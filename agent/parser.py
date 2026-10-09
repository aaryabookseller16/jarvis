import json


def make_arguments(tool_input, params, required):
    """Turn the model's Action Input string into the dict call_tool expects.

    params: the tool's parameter names, in schema order.
    required: the names the tool cannot run without.
    Returns (arguments, None) on success, or (None, error message) so the
    caller can send the message back to the model as a correction.
    """
    # One parameter: the whole string is that parameter, as before.
    if len(params) == 1:
        return {params[0]: tool_input}, None

    # Several parameters: the input must be a JSON object on one line.
    text = tool_input.strip()
    # Models sometimes wrap JSON in a ```json code fence; remove it.
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[len("json"):].strip()

    example = "{" + ", ".join(f'"{p}": ...' for p in required) + "}"
    try:
        arguments = json.loads(text)
    except json.JSONDecodeError:
        return None, (f"Error: this tool takes several inputs, so the Action Input "
                      f"must be a JSON object on one line, for example {example}.")
    if not isinstance(arguments, dict):
        return None, f"Error: the Action Input must be a JSON object, for example {example}."

    unknown = []
    for key in arguments:
        if key not in params:
            unknown.append(key)
    if unknown:
        return None, f"Error: unknown input(s) {unknown}. Valid inputs: {params}."

    missing = []
    for key in required:
        if key not in arguments:
            missing.append(key)
    if missing:
        return None, f"Error: missing required input(s) {missing}. Required: {required}."

    return arguments, None


def parse_input(reply : str):
    # dont bother with the str if it contains the final answer
    if "Final Answer:" in reply:
        answer = reply.split("Final Answer:")[1].strip()
        return ("final", answer)

    lines = reply.splitlines()
    tool = None
    tool_input = None

    for line in lines:
        if line.startswith("Action:"):
            tool = line.split("Action:")[1].strip()
        if line.startswith("Action Input:"):
            tool_input = line.split("Action Input:")[1].strip()

    return ("action", tool, tool_input)

if __name__ == "__main__":
    print(parse_input("Thought: I multiply.\nAction: calculator\nAction Input: 47 * 89"))
    print(parse_input("Thought: Done.\nFinal Answer: 66"))
    print(parse_input("garbage text"))
