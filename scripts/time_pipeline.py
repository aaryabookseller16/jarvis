"""Time each stage of the PDF pipeline on one real PDF.

Usage, from the project root:
    python -m scripts.time_pipeline your.pdf

Runs against a throwaway cache so the first run is genuinely cold and your
real cache/ is untouched. Writes results to cache/timings.json.
"""

import json
import sys
import tempfile
import time

from pathlib import Path

from agent import cache, ollama_client, summarize
from agent.chunker import chunk_text
from agent.tool import PROJECT_ROOT, extract_pdf_text, resolve_in_project


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def main(path):
    full = resolve_in_project(path)

    # 1. the cheap stages, measured on their own
    (text, page_starts), t_extract = timed(extract_pdf_text, full)
    chunks, t_chunk = timed(chunk_text, text, page_starts,
                            cache.CHUNK_SIZE, cache.CHUNK_OVERLAP)

    # 2. load the model first, so load time is not billed to summarization
    _, t_warmup = timed(ollama_client.llm_call, "hello", "Reply with OK.",
                        strict=False)

    # 3. count model calls by wrapping the function summarize.py uses
    calls = {"n": 0}
    real_call = summarize.llm_call

    def counting_call(*args, **kwargs):
        calls["n"] += 1
        return real_call(*args, **kwargs)

    summarize.llm_call = counting_call

    # 4. cold run in a throwaway cache, then a warm run against the same cache
    cache.CACHE_ROOT = Path(tempfile.mkdtemp(prefix="jarvis-timing-"))
    summary, t_cold = timed(summarize.summarize_pdf_cached, full,
                            progress=lambda m: print("  ", m, flush=True))
    cold_calls = calls["n"]
    _, t_warm = timed(summarize.summarize_pdf_cached, full)
    warm_calls = calls["n"] - cold_calls

    results = {
        "file": path,
        "pages": len(page_starts),
        "characters": len(text),
        "chunks": len(chunks),
        "extract_seconds": round(t_extract, 3),
        "chunk_seconds": round(t_chunk, 3),
        "model_warmup_seconds": round(t_warmup, 2),
        "cold_summarize_seconds": round(t_cold, 1),
        "cold_model_calls": cold_calls,
        "seconds_per_call": round(t_cold / max(cold_calls, 1), 2),
        "warm_summarize_seconds": round(t_warm, 3),
        "warm_model_calls": warm_calls,
        "measured_chars_per_token": round(ollama_client.chars_per_token(), 2),
        "summary_characters": len(summary),
    }

    out = PROJECT_ROOT / "cache" / "timings.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m scripts.time_pipeline your.pdf")
    main(sys.argv[1])
