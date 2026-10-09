import json
import re

from agent.cache import cache_path, load_or_build_chunks
from agent.tool import resolve_in_project
TOKEN_RE = re.compile(r"[a-z0-9]+")
TOKENIZER_VERSION = 1

def tokenize(text):
    """Split text into lowercase tokens. Used for chunks AND queries."""
    return TOKEN_RE.findall(text.lower()) # returns list of strings


from collections import Counter
def term_counts(chunks):
    """Per chunk word counts: term_frequencies[i][word] = times word appears in chunk i."""
    term_frequencies = [] # term frequencies
    for c in chunks:
        term_frequencies.append(Counter(tokenize(c["text"])))
    
    return term_frequencies


def doc_freqs(term_frequencies):
    """IDF requires the number of chunks that contain each word"""
    df = Counter()
    for term_frequency in term_frequencies:
        df.update(term_frequency.keys())
    
    return df

def chunk_lengths(term_frequencies):
    """Tokens per chunk: the sum of that chunk's word counts."""
    lengths = []
    for term_frequency in term_frequencies:
        lengths.append(sum(term_frequency.values()))
    
    return lengths


def build_stats(chunks):
    """Everything BM25 needs that does not depend on the query."""
    term_frequencies = term_counts(chunks)
    lengths = chunk_lengths(term_frequencies)
    df = doc_freqs(term_frequencies)

    # avg chunk length
    if len(lengths) > 0:
        avgdl = sum(lengths) / len(lengths)
    else: # if pdf is empty
        avgdl = 0.0
    
    # Convert each Counter to a plain dict so the stats look the same
    # whether freshly built or loaded back from JSON.
    term_frequencies_plain = []
    for term_frequency in term_frequencies:
        term_frequencies_plain.append(dict(term_frequency))

    return {
        "tokenizer_version": TOKENIZER_VERSION,
        "n_chunks": len(chunks),   # N in IDF
        "avgdl": avgdl,            # average length, for length normalization
        "lengths": lengths,        # |D| for each chunk
        "df": dict(df),            # n for each word, for IDF
        "term_frequencies": term_frequencies_plain,          # f: each word's count in each chunk
    }


import math
def idf(n, N):
    """Weight of a word that appears in n of the N chunks.

    Lucene's version of the Robertson/Sparck Jones weight: the 1 inside the
    log keeps it positive even for words in more than half the chunks.
    """
    return math.log(1 + (N - n + 0.5) / (n + 0.5))

K1 = 1.2
B = 0.75
def score_all(query, stats, k1=K1, b =B):
    """Return one BM25 score per chunk, in chunk order."""
    N = stats["n_chunks"]
    avgdl = stats["avgdl"]

    terms = set(tokenize(query))

    scores = []

    for i in range(N):
        term_frequency = stats["term_frequencies"][i] # given chunk word counts
        length = stats["lengths"][i]
        score = 0.0

        for t in terms:
            f = term_frequency.get(t,0) # times t appears in this chunk (0 if absent)
            if f == 0:
                continue
            # Length normalization: above 1 for long chunks, below 1 for short ones
            norm = k1 * (1 - b + b * length / avgdl)
            # IDF times saturated frequency
            score += idf(stats["df"][t], N) * f * (k1 + 1) / (f + norm)
        scores.append(score)

    return scores


DEFAULT_K = 3
def top_k(query, chunks, stats, k = DEFAULT_K):
    """Best k chunks with a score above 0, highest first, as (chunk, score) pairs."""
    scores = score_all(query, stats)

    # Pair each score with its chunk position, so sorting keeps track of
    # which chunk each score belongs to.
    pairs = []
    for i in range(len(chunks)):
        pairs.append((scores[i],i))
    
    # Tuples sort by their first item, so this orders by score, highest first.
    pairs.sort(reverse=True)

    results = []
    for score, i in pairs:
        if len(results) == k:
            break
        if score <= 0:
            break
        
        results.append((chunks[i], score))
    
    return results



def format_results(results):
    """Turn top_k output into the observation text LLM can read."""
    if len(results) == 0:
        return "No passages in this PDF match the query. Try other keywords."

    parts = []
    rank = 1
    for chunk, score in results:
        # "page 4" for a chunk on one page, "pages 4-5" if it spans two
        if chunk["page_start"] == chunk["page_end"]:
            pages = f"page {chunk['page_start']}"
        else:
            pages = f"pages {chunk['page_start']}-{chunk['page_end']}"

        parts.append(f"[{rank}] {pages} (score {score:.2f})\n{chunk['text'].strip()}")
        rank += 1

    # A blank line between results so qwen can tell where each one ends
    return "\n\n".join(parts)


def load_or_build_stats(requested_path):
    """Return (chunks, stats) for a PDF, using cached bm25.json when it is still valid."""
    # Chunks first: this reuses (or rebuilds) chunks.json via cache.py.
    chunks = load_or_build_chunks(requested_path)

    # Same folder as chunks.json: cache/<sha256 of the PDF>/bm25.json
    stats_file = cache_path(resolve_in_project(requested_path)) / "bm25.json"

    if stats_file.exists():
        try:
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
            same_tokenizer = stats.get("tokenizer_version") == TOKENIZER_VERSION
            same_chunks = stats.get("n_chunks") == len(chunks)
            if same_tokenizer and same_chunks:
                return chunks, stats        # cache hit
        except (json.JSONDecodeError, OSError):
            pass                            # unreadable file: rebuild below

    # Cache miss or stale: build, save, return.
    stats = build_stats(chunks)
    stats_file.write_text(json.dumps(stats), encoding="utf-8")
    return chunks, stats


# run the whole pipeline
MAX_K = 5
def search_pdf(requested_path, query, k=DEFAULT_K):
    """Cached stats, score every chunk, keep the top k, format for qwen."""
    # qwen chooses k, so keep it in a safe range: at least 1, at most MAX_K.
    k = int(k)
    if k < 1:
        k = 1
    if k > MAX_K:
        k = MAX_K

    chunks, stats = load_or_build_stats(requested_path)
    results = top_k(query, chunks, stats, k)
    return format_results(results)