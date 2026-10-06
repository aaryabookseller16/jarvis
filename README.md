# Jarvis

A local ReAct agent built from scratch with Ollama, no agent frameworks.

## Layout

- `agent/chat.py` — the client: runs the chat loop, prompts the model, and parses
  its ReAct-style replies (`agent/parser.py`). It launches the MCP server as a
  subprocess over stdio, discovers its tools at startup, and calls them through
  an MCP `ClientSession`.
- `agent/server.py` — the MCP server exposing the tools.
- `agent/tool.py` — the tool implementations (calculator, read_file,
  list_directory, search_files), all sandboxed to the project directory:
  paths and search patterns that point outside it are refused, and
  search_files skips `venv/`, `.git/`, `__pycache__/` and `cache/`.
- `agent/chunker.py` — text chunking (greedy packing / recursive splitting),
  producing chunks that carry the pages they came from.
- `agent/cache.py` — per PDF cache at `cache/<sha256>/`, with a `manifest.json`
  recording the settings each artifact was built with.
- `agent/ollama_client.py` — the only module that calls Ollama. Sets `num_ctx`
  explicitly and raises on suspected prompt truncation.
- `agent/summarize.py` — map reduce summarization over the cached chunks.
- `scripts/time_pipeline.py` — times each stage of the PDF pipeline (extract,
  chunk, cold and warm summarize) on one real PDF, using a throwaway cache.
- `tests/` — `test_tools.py` tool sandbox and parser checks, `test_chunker.py`
  chunker checks, `test_cache.py` cache and summarizer checks (Ollama and the PDF extractor are faked, so it needs
  neither a model nor pypdf), and `smoke_client.py`, a standalone script that
  exercises the server directly without the chat loop.

## Usage

Run from the project root, with Ollama running (`ollama serve`) and
`qwen2.5:14b` pulled:

```
pip install -r requirements.txt
python -m agent.chat
python tests/smoke_client.py
python -m tests.test_chunker
python -m tests.test_cache
python -m tests.test_tools
python -m scripts.time_pipeline your.pdf
```

Work in progress.
