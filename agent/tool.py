"""Calculator tool for the ReAct agent.

Safely evaluates arithmetic expression strings (e.g. "47 * 89", "(3 + 4) * 2")
WITHOUT using eval(). It parses the string into an AST and walks it, permitting
only arithmetic node types. Anything else — function calls, names, attribute
access, imports — raises ValueError instead of executing.

This replaces the earlier `return eval(action_input)` placeholder, which would
run arbitrary Python from the model's output.
"""

import ast
import operator
from pathlib import Path

# Tools are sandboxed to the project root (one level above this package).
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Allowed binary operators: AST node type -> function that performs it.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Larger exponents can take minutes to compute and block the server.
MAX_EXPONENT = 1000

# Allowed unary operators (e.g. the minus in -5).
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node):
    """Recursively evaluate a single AST node, allowing only arithmetic."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    # A literal number, e.g. 47 or 3.14
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")

    # A binary operation, e.g. 47 * 89
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent too large (limit {MAX_EXPONENT})")
        return _BIN_OPS[op_type](left, right)

    # A unary operation, e.g. -5
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))

    # Anything else (Name, Call, Attribute, etc.) is rejected.
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculator(action_input):
    """Evaluate an arithmetic expression string and return the result as text.

    Returns a string on error so the agent loop can feed it back to the model
    as an Observation instead of crashing.
    """
    try:
        tree = ast.parse(action_input, mode="eval")
        # str() inside the try: a huge integer can fail to convert to text.
        return str(_eval_node(tree))
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError) as e:
        return f"Error: could not evaluate {action_input!r} ({e})"

def read_file(requested_path):
    # 1. allowed dir as a resolved Path
    ALLOWED_DIRECTORY = PROJECT_ROOT

    # 2. join with requested_path, resolve it
    try:
        full_path =  ALLOWED_DIRECTORY / Path(requested_path)
        file = full_path.resolve()
    except (TypeError, ValueError, OSError, RuntimeError):
        return f"Please submit a file inside {ALLOWED_DIRECTORY}"

    # 3. if file is NOT inside allowed dir -> return a refusal string
    if not file.is_relative_to(ALLOWED_DIRECTORY):
        return f"Please submit a file inside {ALLOWED_DIRECTORY}"

    # 4. if it doesn't exist or isn't a regular file -> return a "not found" string
    try:
        if not file.is_file():
            return f"This file does not exist in {ALLOWED_DIRECTORY}"
    except OSError:
        return f"This file does not exist in {ALLOWED_DIRECTORY}"
    if file.suffix.lower() == ".pdf":
        return "that file is a PDF; use summarize_pdf instead"
     # 5. else: read the text and return it, inside try/except so a
     # permission or decode error returns a message instead of crashing
    else:
        try:
            llm_input = file.read_text()
            return llm_input
        except PermissionError:
            return "no permission to read that file"
        except UnicodeDecodeError:
            return "that file isn't readable"
        except OSError:
            return "that file isn't readable"

    return ""

def list_directory(requested_path="."):
    """List entries in a directory inside the project. Read only, allow-listed."""
    ALLOWED_DIRECTORY = PROJECT_ROOT
    try:
        full = (ALLOWED_DIRECTORY / Path(requested_path)).resolve()
    except (TypeError, ValueError, OSError, RuntimeError):
        return f"Please submit a path inside {ALLOWED_DIRECTORY}"
    if not full.is_relative_to(ALLOWED_DIRECTORY):
        return f"Please submit a path inside {ALLOWED_DIRECTORY}"
    try:
        if not full.is_dir():
            return f"{requested_path} is not a directory inside {ALLOWED_DIRECTORY}"
        entries = []
        for p in sorted(full.iterdir()):
            kind = "dir" if p.is_dir() else "file"
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            entries.append(f"{kind}\t{size}\t{p.name}")
    except OSError:
        return "that directory isn't readable"
    return "\n".join(entries) if entries else "(empty directory)"


# Folders search_files never descends into: environment, VCS and generated
# files, which would bury the project's own files and overflow the context.
SEARCH_SKIP_DIRS = {"venv", ".git", "__pycache__", "cache"}
MAX_SEARCH_RESULTS = 200


def search_files(pattern):
    """Find files by name or glob pattern recursively inside the project. Read only, allow-listed."""
    ALLOWED_DIRECTORY = PROJECT_ROOT
    if not isinstance(pattern, str) or not pattern.strip():
        return "Please provide a filename pattern, for example '*.py'"
    # Matching is lexical, so "../*" would walk out of the project. Refuse it
    # up front instead of filtering results afterwards.
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        return f"Please submit a pattern inside {ALLOWED_DIRECTORY}"
    matches = []
    try:
        for p in ALLOWED_DIRECTORY.rglob(pattern):
            try:
                rel = p.relative_to(ALLOWED_DIRECTORY)
                # resolve() follows symlinks, which could point outside.
                if not p.resolve().is_relative_to(ALLOWED_DIRECTORY):
                    continue
            except (ValueError, OSError):
                continue
            if SEARCH_SKIP_DIRS.intersection(rel.parts):
                continue
            matches.append(str(rel))
    except (OSError, ValueError, NotImplementedError) as e:
        return f"Error searching for {pattern!r} ({e})"
    if not matches:
        return f"No files matching {pattern!r}"
    matches.sort()
    if len(matches) > MAX_SEARCH_RESULTS:
        extra = len(matches) - MAX_SEARCH_RESULTS
        matches = matches[:MAX_SEARCH_RESULTS] + [f"... {extra} more, narrow the pattern"]
    return "\n".join(matches)


class PdfError(Exception):
    """Raised when a PDF can't be turned into text.

    The message is written for the model to read, so the MCP wrapper can
    return str(e) directly as the tool result.
    """


def resolve_in_project(requested_path):
    """Resolve requested_path to a real file inside the project, or raise PdfError.

    One place for the allow list check, so every caller (extractor, cache,
    summarizer) enforces the same boundary. An absolute path that already
    points inside the project is accepted, because pathlib's / operator
    returns the absolute path unchanged.
    """
    ALLOWED_DIRECTORY = PROJECT_ROOT
    try:
        full = (ALLOWED_DIRECTORY / Path(requested_path)).resolve()
    except (TypeError, ValueError, OSError, RuntimeError):
        raise PdfError(f"Please submit a file inside {ALLOWED_DIRECTORY}")
    if not full.is_relative_to(ALLOWED_DIRECTORY):
        raise PdfError(f"Please submit a file inside {ALLOWED_DIRECTORY}")
    if not full.is_file():
        raise PdfError(f"This file does not exist in {ALLOWED_DIRECTORY}")
    return full


def extract_pdf_text(requested_path):
    """Extract text from a PDF inside the project. Read only, allow-listed.

    Returns (text, page_starts):
        text: every page's text joined with "\\n". Not stripped, so the
            offsets below stay valid.
        page_starts: page_starts[i] is the character offset in `text` where
            page i (0 indexed) begins. page_starts[0] is always 0.

    To find the page of character position p:
        bisect.bisect_right(page_starts, p) - 1   (0 indexed)

    Raises PdfError for a path outside the project, a missing file, a
    missing pypdf install, a locked or corrupt PDF, or a PDF with no
    extractable text (for example scanned images).
    """
    full = resolve_in_project(requested_path)

    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError:
        raise PdfError("pypdf is not installed; run: pip install pypdf")

    try:
        reader = PdfReader(str(full))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise PdfError("that PDF is encrypted and needs a password")
            except PdfError:
                raise
            except Exception:
                raise PdfError("that PDF is encrypted and needs a password")

        pages = []
        page_starts = []
        offset = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            page_starts.append(offset)
            pages.append(page_text)
            offset += len(page_text) + 1  # +1 for the "\n" join separator
        text = "\n".join(pages)
    except (PdfReadError, OSError, ValueError) as e:
        raise PdfError(f"could not read that PDF ({e})") from e

    if not text.strip():
        raise PdfError("that PDF has no extractable text (it may be scanned images)")
    return text, page_starts


if __name__ == "__main__":
    print(read_file("tests/fixtures/todo.txt"))
    print(read_file("../secrets.txt"))
    print(read_file("/etc/passwd"))
    print(read_file("does_not_exist.txt"))
