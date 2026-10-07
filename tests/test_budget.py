"""fit_to_budget tests. Pure list logic: no Ollama, no database.

Run: python -m tests.test_budget
"""

from agent import chat
from agent.chat import fit_to_budget


def msg(role, kind, chars):
    return {"role": role, "kind": kind, "content": "x" * chars}


def turn(chars=100, tool=True):
    """One question, optionally a tool call + observation, then a final answer."""
    rows = [msg("user", "user_input", chars)]
    if tool:
        rows += [msg("assistant", "tool_call", chars), msg("user", "observation", chars)]
    return rows + [msg("assistant", "final_answer", chars)]


def main():
    system = {"role": "system", "content": "s" * 200}
    chat.TOKEN_BUDGET = 1000
    chat.CHARS_PER_TOKEN = 2

    # 1. everything fits: nothing dropped, input list untouched
    messages = [system] + turn() + turn() + [msg("user", "user_input", 50)]
    before = list(messages)
    assert fit_to_budget(messages) == messages
    assert messages == before, "fit_to_budget changed its input"
    print("OK small history passes through unchanged")

    # 2. over budget: oldest turns dropped, system prompt kept first
    messages = [system] + turn(300) * 5 + [msg("user", "user_input", 50)]
    out = fit_to_budget(messages)
    assert out[0] is system
    assert len(out) < len(messages)
    assert out[-1] is messages[-1]
    used = sum(len(m["content"]) for m in out) // 2
    assert used <= chat.TOKEN_BUDGET, used
    print(f"OK trimmed {len(messages)} -> {len(out)} messages, ~{used} tokens")

    # 3. never starts mid turn: first kept history row is a question
    for big in (150, 333, 400, 777):
        messages = [system] + turn(big) * 4 + [msg("user", "user_input", 10)]
        assert fit_to_budget(messages)[1]["kind"] == "user_input", big
    print("OK trimmed history always starts on a user_input")

    # 4. huge current turn: kept whole, all older history dropped
    current = [msg("user", "user_input", 10), msg("assistant", "tool_call", 10),
               msg("user", "observation", 50_000)]
    messages = [system] + turn() + current
    assert fit_to_budget(messages) == [system] + current
    print("OK oversized current turn kept, older history dropped")

    print("all budget tests passed")


if __name__ == "__main__":
    main()
