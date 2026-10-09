import asyncio          # runs the async code (the event loop + asyncio.run)
import sys              # gives us sys.executable, the path to the current Python

import ollama           # talks to the local model
from mcp import ClientSession, StdioServerParameters, stdio_client  # the MCP client pieces

from agent import db
from agent.parser import make_arguments, parse_input

MODEL = "qwen2.5:14b"
NUM_CTX = 8192   # Ollama defaults to 4096 and silently truncates past it
MAX_STEPS = 10   # tool calls allowed per question, so a model stuck calling tools still stops
TOKEN_BUDGET = 6000   # prompt tokens we allow, leaving the rest of NUM_CTX for the reply
CHARS_PER_TOKEN = 2   # conservative: measured 1.94 to 4.66 chars per token
HISTORY_ROWS = 200    # rows loaded at startup; fit_to_budget decides what the model sees

# The fixed part of the system prompt: HOW to reply. This is scaffolding, not tool
# info, so it is hardcoded. The tool list itself is added dynamically below.
FORMAT_RULES = """
Respond in EXACTLY this format if you use a tool, then wait for the tool's answer.
Thought: <your reasoning about what to do next>
Action: <the exact name of one tool from the list above>
Action Input: <the input to that tool: plain text if the tool takes one input, or a JSON object on one line if it takes several>

When you have the final answer, respond in EXACTLY this format:
Thought: <your reasoning>
Final Answer: <your answer>

Produce only ONE Thought and ONE Action, then stop. Do NOT write an Observation yourself; it will be provided to you.

If the message does not need a tool, such as a greeting or small talk, skip the Action and reply directly using the Thought/Final Answer format.
"""


def build_system_prompt(tools):
    # Build the tool menu from what the server reported, instead of typing it by hand.
    # Each tool contributes one line: its name and the description from its docstring.
    lines = ["You are an agent that solves problems step by step using tools.", "", "Available tools:"]
    for t in tools:
        params = list(t.input_schema["properties"])
        required = t.input_schema.get("required", [])
        if len(params) == 1:
            lines.append(f"- {t.name}: {t.description}")
        else:
            # Several inputs: name them, so the model knows the JSON keys to use.
            names = []
            for p in params:
                names.append(p if p in required else f"{p} (optional)")
            lines.append(f"- {t.name}: {t.description} "
                         f"Inputs: {', '.join(names)}. Action Input must be a JSON object.")
    # Menu on top, then the format rules underneath.
    return "\n".join(lines) + "\n" + FORMAT_RULES


def estimate_tokens(message):
    return len(message["content"]) // CHARS_PER_TOKEN


def fit_to_budget(messages):
    # Return a trimmed COPY for the model: the system prompt plus the newest
    # messages that fit in TOKEN_BUDGET. `messages` itself is never changed, so
    # turn_start in main() stays valid.
    system, history = messages[0], messages[1:]
    # The current turn (from the last user_input on) is always kept, even over
    # budget, or the model would not see the question.
    current = max(i for i, m in enumerate(history) if m.get("kind") == "user_input")
    used = estimate_tokens(system) + sum(estimate_tokens(m) for m in history[current:])
    keep = current
    for i in range(current - 1, -1, -1):
        used += estimate_tokens(history[i])
        if used > TOKEN_BUDGET:
            break
        keep = i
    # Start on a question, never mid turn (an orphan observation or answer).
    while history[keep].get("kind") != "user_input":
        keep += 1
    return [system] + history[keep:]


def ask_model(messages):
    # One model call with the settings every call shares. Returns the reply text.
    # Raises ConnectionError when Ollama is not running, ollama.ResponseError on
    # a server side problem such as a missing model.
    # Send only role and content: `kind` is our bookkeeping, not Ollama's.
    payload = [{"role": m["role"], "content": m["content"]} for m in fit_to_budget(messages)]
    resp = ollama.chat(model=MODEL, messages=payload, options={"temperature": 0, "num_ctx": NUM_CTX})
    return resp["message"]["content"]


async def run_tool(session, name, arguments):
    # Run one tool ON THE SERVER and return its text output as the observation.
    # A failure becomes an error string the model can read, instead of a crash.
    try:
        result = await session.call_tool(name, arguments)
    except Exception as e:
        return f"Error: tool {name} failed ({e})"
    # The tool's text output is the first content block's text.
    if not result.content:
        return f"Error: tool {name} returned nothing"
    return result.content[0].text


def remember(conn, conv_id, messages, role, kind, content):
    # Keep the in-memory list and the database in step: one call does both.
    messages.append({"role": role, "kind": kind, "content": content})
    db.insert_message(conn, conv_id, role, kind, content)


async def main():
    # Recipe for starting the server: run "python -m agent.server" with THIS same Python,
    # so the subprocess uses your venv. Nothing launches yet; this is just the plan.
    server_params = StdioServerParameters(command=sys.executable, args=["-m", "agent.server"])

    # stdio_client launches the server as a subprocess and hands back the two pipe ends.
    # async with keeps the ONE server alive for the whole chat and cleans it up on exit.
    async with stdio_client(server_params) as (read, write):
        # Wrap the raw pipe in a session so we can speak in tools, not bytes.
        async with ClientSession(read, write) as session:
            # Handshake. Must happen before any other call.
            await session.initialize()

            # Ask the server what tools exist. .tools is the list of tool definitions.
            tool_list = (await session.list_tools()).tools

            # Each tool's parameter names and which of them are required, read from its
            # input schema. Tools with one parameter take the Action Input as plain text;
            # tools with several take a JSON object (see make_arguments in parser.py).
            tool_params = {}
            for t in tool_list:
                params = list(t.input_schema["properties"])
                required = t.input_schema.get("required", [])
                tool_params[t.name] = (params, required)

            # The set of tool names the server actually offers, used to validate the
            # model's chosen Action below.
            valid_tools = set(tool_params)

            conn = db.connect()
            conv_id = db.latest_or_new_conversation(conn)
            # Fresh system prompt (decision 5) + saved history.
            messages = [{"role": "system", "content": build_system_prompt(tool_list)}]
            messages += db.load_recent(conn, conv_id, HISTORY_ROWS)

            # Main chat loop: one pass per user question.
            while True:
                # Ctrl-D (EOF) or Ctrl-C at the prompt ends the chat cleanly.
                try:
                    user = input("> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    conn.close()
                    return
                if not user.strip():
                    continue

                # Remember where this turn starts, so a failed turn can be undone.
                turn_start = len(messages)
                remember(conn, conv_id, messages, "user", "user_input", user)

                try:
                    failed = await run_turn(session, messages, tool_params, valid_tools, conn, conv_id)
                except ConnectionError:
                    print("Ollama is not running; start it with `ollama serve` and ask again.")
                    failed = True
                except ollama.ResponseError as e:
                    print(f"Ollama error: {e.error}")
                    failed = True

                # A turn that never reached a Final Answer leaves a half finished
                # exchange; drop it so the next question starts from clean history.
                if failed:
                    del messages[turn_start:]
                    conn.rollback()
                else:
                    conn.commit()


async def run_turn(session, messages, tool_params, valid_tools, conn, conv_id):
    # One user question: loop model -> tool -> observation until a Final Answer.
    # Returns True if the turn failed, False if it ended with a Final Answer.

    # Ask the model. This is where the model decides: use a tool, or answer.
    # It is a normal blocking call (the ollama library is synchronous).
    reply = ask_model(messages)
    reply_tuple = parse_input(reply)   # ("final", answer) or ("action", name, input)

    retries = 3        # how many malformed replies we tolerate before giving up
    steps = 0          # how many tools have run this turn
    failed = False

    # Keep looping as long as the model wants a tool (not a final answer).
    while reply_tuple[0] != "final":
        tool_name = reply_tuple[1]
        tool_input = reply_tuple[2]

        # Turn the Action Input into the tool's arguments dict. arg_error is set
        # when the input does not fit the tool (bad JSON, missing inputs).
        arguments, arg_error = None, None
        if tool_name in valid_tools and tool_input is not None:
            params, required = tool_params[tool_name]
            arguments, arg_error = make_arguments(tool_input, params, required)

        if arguments is not None:
            # Too many tool calls means the model is going in circles: stop.
            if steps == MAX_STEPS:
                print(f"Stopped: more than {MAX_STEPS} tool calls for one question.")
                failed = True
                break
            call_kind, obs_kind = "tool_call", "observation"
            steps += 1
            # Valid tool + input: build the arguments dict keyed by the
            # tool's real parameter name, then run it ON THE SERVER.
            # This await waits for the server to execute the tool, not the model.
            observation = await run_tool(session, tool_name, arguments)
        else:
            # Something was wrong with the model's reply. Figure out which
            # failure it was and build a correction to send back as feedback.
            if arg_error is not None:
                # Right tool, but the input did not fit its parameters.
                correction = arg_error
            elif tool_name is None and tool_input is None:
                # Neither an Action nor a Final Answer: reply was off-format.
                correction = (
                    "Error: your reply had no 'Action:' line and no 'Final Answer:' line. "
                    "Respond in EXACTLY one of the two formats: "
                    "'Thought:/Action:/Action Input:' to use a tool, or "
                    "'Thought:/Final Answer:' when you are done."
                )
            elif tool_name not in valid_tools:
                # It named a tool that does not exist.
                correction = (
                    f"Error: '{tool_name}' is not a valid tool. "
                    f"Available tools: {sorted(valid_tools)}."
                )
            else:
                # It gave an Action but forgot the Action Input line.
                correction = "Error: you emitted an Action but no 'Action Input:' line. Include it."

            # Out of retries: stop trying to salvage this turn.
            if retries == 0:
                print("Failed: the model couldn't produce a valid tool call.")
                failed = True
                break
            call_kind, obs_kind = "failed_tool_call", "correction"
            # Otherwise, the correction becomes the observation and we spend a retry.
            observation = correction
            retries -= 1

        # Feed the observation (tool result OR correction) back to the model
        # so it can decide the next step with the new information.
        remember(conn, conv_id, messages, "assistant", call_kind, reply)
        remember(conn, conv_id, messages, "user", obs_kind, f"Observation: {observation}")

        # Ask the model again, now that it has seen the observation.
        reply = ask_model(messages)
        reply_tuple = parse_input(reply)

    # Loop ended because the model gave a Final Answer (and we did not fail).
    if not failed:
        print(reply)
        remember(conn, conv_id, messages, "assistant", "final_answer", reply)
    return failed


if __name__ == "__main__":
    # Kick off the whole async chain. main() alone would only create a coroutine;
    # asyncio.run actually drives it to completion.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass   # Ctrl-C while the model or a tool is working: exit quietly
