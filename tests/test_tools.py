"""Tool and parser checks. No Ollama and no MCP server needed.

Run: python -m tests.test_tools
"""

import time

from agent.parser import parse_input
from agent.tool import (MAX_SEARCH_RESULTS, PROJECT_ROOT, calculator, list_directory,
                        read_file, search_files)

OUTSIDE = f"inside {PROJECT_ROOT}"   # every refusal names the allowed directory


def main():
    # 1. calculator: arithmetic works, everything else is an error string
    assert calculator("47 * 89") == "4183"
    assert calculator("(3 + 4) * 2") == "14"
    assert calculator("-2 ** 3") == "-8"
    assert calculator("1/0").startswith("Error")
    assert calculator("__import__('os')").startswith("Error")
    assert calculator("47 *").startswith("Error")
    print("OK calculator")

    # 2. calculator: huge powers fail fast instead of hanging or crashing
    for expr in ["9**9**9", "2**2**2**2**2", "10.0 ** 1000", "(10**1000)**1000"]:
        start = time.perf_counter()
        result = calculator(expr)
        assert result.startswith("Error"), f"{expr} -> {result[:80]}"
        assert time.perf_counter() - start < 2, f"{expr} took too long"
    print("OK calculator limits")

    # 3. read_file stays inside the project and points PDFs at summarize_pdf
    assert read_file("tests/fixtures/todo.txt").startswith("hello")
    for path in ["../x", "/etc/passwd", "tests/../../x"]:
        assert OUTSIDE in read_file(path), path
    assert "does not exist" in read_file("agent")
    assert "summarize_pdf" in read_file("tests/fixtures/fake_doc.pdf")
    print("OK read_file")

    # 4. list_directory stays inside the project
    assert "todo.txt" in list_directory("tests/fixtures")
    for path in ["..", "/etc", "agent/../.."]:
        assert OUTSIDE in list_directory(path), path
    assert "not a directory" in list_directory("README.md")
    print("OK list_directory")

    # 5. search_files refuses patterns that leave the project
    for pattern in ["../*", "**/../../*", "/etc/*", "agent/../../*"]:
        assert OUTSIDE in search_files(pattern), pattern
    assert search_files("").startswith("Please provide")
    print("OK search_files sandbox")

    # 6. search_files skips venv, .git, caches, and caps its output
    found = search_files("*.py").splitlines()
    assert "agent/chat.py" in found
    assert not any(p.startswith(("venv/", ".git/")) or "__pycache__" in p
                   for p in found), "searched a skipped folder"
    assert len(found) <= MAX_SEARCH_RESULTS + 1
    assert search_files("no_such_file_*.xyz").startswith("No files matching")
    print("OK search_files results")

    # 7. parser: actions, final answers, and off format replies
    assert parse_input("Thought: x\nAction: calculator\nAction Input: 47 * 89") == \
        ("action", "calculator", "47 * 89")
    assert parse_input("Thought: Done.\nFinal Answer: 66") == ("final", "66")
    assert parse_input("garbage text") == ("action", None, None)
    assert parse_input("Thought: x\nAction: calculator") == ("action", "calculator", None)
    print("OK parser")

    print("all tool tests passed")


if __name__ == "__main__":
    main()
