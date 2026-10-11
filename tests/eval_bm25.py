"""Recall@k for BM25 search_pdf on a fixed question set.

A question is a hit at k if any of the top k chunks contains its answer text.
Questions with answer null (the PDF does not cover them) are shown but not
scored. Needs no Ollama: it only reads the cached chunks and BM25 stats.

Run: python -m tests.eval_bm25 [tests/fixtures/eval_sample_pdf.json]
"""

import json
import sys

from agent.bm25 import load_or_build_stats, top_k

KS = [1, 3, 5]


def first_hit_rank(results, answer):
    """1 based rank of the first result containing answer, or None."""
    rank = 1
    for chunk, score in results:
        if answer.lower() in chunk["text"].lower():
            return rank
        rank += 1
    return None


def main(path):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    chunks, stats = load_or_build_stats(spec["pdf"])

    hits = {}
    for k in KS:
        hits[k] = 0
    scored = 0
    by_type = {}

    for q in spec["questions"]:
        results = top_k(q["question"], chunks, stats, max(KS))
        pages = []
        for chunk, score in results:
            pages.append(f"p{chunk['page_start']}" if chunk["page_start"] == chunk["page_end"]
                         else f"p{chunk['page_start']}-{chunk['page_end']}")

        if q["answer"] is None:
            print(f"{q['id']:4} {q['type']:11} not scored   top5 {pages}  {q['question']}")
            continue

        rank = first_hit_rank(results, q["answer"])
        scored += 1
        for k in KS:
            if rank is not None and rank <= k:
                hits[k] += 1
        t = q["type"]
        if t not in by_type:
            by_type[t] = [0, 0]
        by_type[t][1] += 1
        if rank is not None and rank <= 3:
            by_type[t][0] += 1
        shown = rank if rank is not None else "miss"
        print(f"{q['id']:4} {q['type']:11} rank {str(shown):5}  top5 {pages}  {q['question']}")

    print()
    for k in KS:
        print(f"recall@{k}: {hits[k]}/{scored} = {hits[k] / scored:.2f}")
    print("recall@3 by type:")
    for t in by_type:
        print(f"  {t:11} {by_type[t][0]}/{by_type[t][1]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("tests/fixtures/eval_sample_pdf.json")
