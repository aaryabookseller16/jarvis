"""Per PDF cache of derived artifacts, keyed by the file's content hash.

The hash identifies the INPUT: rename the file and the cache still hits,
edit the file and the cache misses. The manifest records the SETTINGS that
produced each artifact, so changing chunk size, a model, or a prompt
invalidates what those settings produced even though the PDF is unchanged.
Input identity and output validity are different questions.
"""

import hashlib
import json

from pathlib import Path

from agent.chunker import chunk_text
from agent.tool import PROJECT_ROOT, extract_pdf_text, resolve_in_project

CACHE_ROOT = PROJECT_ROOT / "cache"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MANIFEST_VERSION = 1


def file_sha256(path, block_size=1 << 20):
    """Hash a file's bytes in blocks, so a large PDF never loads into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_path(full_path):
    """Return cache/<sha256>/ for an already resolved file, without creating it."""
    return CACHE_ROOT / file_sha256(full_path)


def cache_dir(full_path):
    """Return (and create) cache/<sha256>/ for an already resolved file."""
    directory = cache_path(full_path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_manifest(directory):
    """Read manifest.json, treating a missing or corrupt one as empty."""
    path = directory / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}  # a broken manifest means rebuild, never crash


def write_manifest(directory, updates):
    """Merge updates into manifest.json and return the merged manifest."""
    manifest = read_manifest(directory)
    manifest.update(updates)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_or_build_chunks(requested_path):
    """Return the chunk list for a PDF, building and caching it when stale."""
    full = resolve_in_project(requested_path)
    directory = cache_path(full)   # created only once extraction succeeds
    manifest = read_manifest(directory)
    chunks_file = directory / "chunks.json"

    fresh = (
        chunks_file.exists()
        and manifest.get("manifest_version") == MANIFEST_VERSION
        and manifest.get("chunk_size") == CHUNK_SIZE
        and manifest.get("chunk_overlap") == CHUNK_OVERLAP
    )
    if fresh:
        try:
            return json.loads(chunks_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # unreadable cache: fall through and rebuild

    text, page_starts = extract_pdf_text(full)
    directory.mkdir(parents=True, exist_ok=True)
    chunks = chunk_text(text, page_starts, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunks_file.write_text(json.dumps(chunks), encoding="utf-8")

    # Anything derived FROM these chunks is now stale.
    (directory / "summary.txt").unlink(missing_ok=True)
    (directory / "bm25.json").unlink(missing_ok=True)
    write_manifest(directory, {
        "manifest_version": MANIFEST_VERSION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "n_chunks": len(chunks),
        "n_pages": len(page_starts),
        "summary_model": None,
        "summary_prompt_version": None,
    })
    return chunks
