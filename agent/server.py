import logging

from mcp.server import MCPServer
from agent.tool import calculator as _calculator, read_file as _read_file, list_directory as _list_directory, search_files as _search_files
from agent.tool import PdfError
from agent.summarize import summarize_pdf_cached

# pypdf logs warnings about malformed PDFs to stderr, which the chat client
# shares, so they would print over the prompt. The tool result already says
# what went wrong.
logging.getLogger("pypdf").setLevel(logging.ERROR)

mcp = MCPServer("jarvis-tools")

@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, for example '47 * 89' or '(3 + 4) * 2'. Use this for any math."""
    return _calculator(expression)

@mcp.tool()
def read_file(requested_path: str) -> str:
    """Read a text file inside the project directory. Pass a path relative to the project, for example 'README.md'."""
    return _read_file(requested_path)

@mcp.tool()
def list_directory(path: str = ".") -> str:
    """List files and folders inside the project directory. Pass a path relative to the project, or '.' for the project root. Returns type, size, and name per entry."""
    return _list_directory(path)

@mcp.tool()
def search_files(pattern: str) -> str:
    """Find files by name or glob pattern (for example '*.py' or 'READ*') recursively inside the project directory. Returns matching paths relative to the project."""
    return _search_files(pattern)


@mcp.tool()
def summarize_pdf(path: str) -> str:
    """Summarize a PDF inside the project directory. Pass a path relative to the project, for example 'paper.pdf'. Use this instead of read_file for PDFs and for any document too long to read in full. The first call on a PDF is slow because it reads the whole document; later calls are cached."""
    try:
        return summarize_pdf_cached(path)
    except PdfError as e:
        return str(e)
    except ConnectionError:
        # What the ollama client raises when no server is listening.
        return "Ollama is not running; start it with `ollama serve`"


if __name__ == "__main__":
    mcp.run(transport="stdio")
