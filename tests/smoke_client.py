import asyncio
import sys

from mcp import ClientSession, StdioServerParameters, stdio_client


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,   # same Python as this script, so the venv is used
        args=["-m", "agent.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools the server offers:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description}")

            calls = [
                ("calculator", {"expression": "47 * 89"}),
                ("read_file", {"requested_path": "tests/fixtures/todo.txt"}),
                ("list_directory", {"path": "tests/fixtures"}),
                ("search_files", {"pattern": "*.py"}),
                # A path outside the project: answers with a refusal, needs no Ollama.
                ("summarize_pdf", {"path": "../outside.pdf"}),
                # BM25 search on the sample PDF: needs no Ollama, uses cached stats.
                ("search_pdf", {"path": "sample.pdf", "query": "contract design", "k": 2}),
            ]
            for name, arguments in calls:
                result = await session.call_tool(name, arguments)
                print(f"{name}({arguments}) ->", result.content[0].text[:200])

if __name__ == "__main__":
    asyncio.run(main())