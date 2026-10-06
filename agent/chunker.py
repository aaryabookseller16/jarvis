"""Split extracted PDF text into overlapping chunks that carry page numbers.

The whole module works with spans, meaning (start, end) pairs of character
offsets into the original text, with end exclusive. Text is only sliced out
at the very end, in chunk_text. Carrying offsets instead of strings is what
lets every chunk report the pages it came from, and it keeps repeated text
(headers, boilerplate) from being confused with itself.

Sizes are measured in CHARACTERS, not tokens. For English, tokens are roughly
characters / 4, so size=2000 is about 500 tokens. That ratio is an estimate;
check it against a real PDF before trusting a context budget.
"""

from bisect import bisect_right

# Tried in order. Earlier separators are more semantic: paragraph, line,
# sentence, word. The last resort is a hard cut at exactly `size`.
SEPARATORS = ["\n\n", "\n", ". ", " "]


def split_recursive(text, start, end, size, seps=SEPARATORS):
    """Split text[start:end] into contiguous spans, each at most `size` long.

    Splits on the first separator in `seps` that occurs in the range. Any
    piece that is still too long is split again with the remaining, finer
    separators. If no separator works, the range is cut at fixed offsets.

    The pieces are contiguous and lossless: they cover [start, end) exactly,
    with each separator kept at the end of the piece before it.
    """
    if end - start <= size:
        return [(start, end)]

    for i, sep in enumerate(seps):
        segments = []
        cur = start
        idx = text.find(sep, cur, end)
        while idx != -1:
            cut = idx + len(sep)
            if cut > end:
                break
            segments.append((cur, cut))
            cur = cut
            idx = text.find(sep, cur, end)
        if cur < end:
            segments.append((cur, end))

        if len(segments) <= 1:
            continue  # this separator does not actually divide the range

        pieces = []
        for s, e in segments:
            if e - s <= size:
                pieces.append((s, e))
            else:
                pieces.extend(split_recursive(text, s, e, size, seps[i + 1:]))
        return pieces

    # No separator divided the range: cut at fixed character offsets.
    return [(s, min(s + size, end)) for s in range(start, end, size)]


def pack_greedy(spans, size, overlap):
    """Merge contiguous spans into chunk spans of at most `size` characters.

    Greedy: keep adding pieces until the next one would overflow, then close
    the chunk and restart from its trailing pieces, up to `overlap`
    characters of them. The overlap is dropped whenever keeping it would
    overflow the next chunk or stall the loop, so progress is guaranteed.
    """
    chunks = []
    cur = []  # pieces in the chunk currently being built

    for sp in spans:
        if cur and sp[1] - cur[0][0] > size:
            chunks.append((cur[0][0], cur[-1][1]))

            tail = []
            total = 0
            for piece in reversed(cur):
                length = piece[1] - piece[0]
                if total + length > overlap:
                    break
                tail.insert(0, piece)
                total += length

            # Re-using every piece would rebuild the same chunk forever.
            if tail and tail[0][0] == cur[0][0]:
                tail = []
            # Overlap must not push the new chunk over the size limit.
            if tail and sp[1] - tail[0][0] > size:
                tail = []
            cur = tail

        cur.append(sp)

    if cur:
        chunks.append((cur[0][0], cur[-1][1]))
    return chunks


def page_of(pos, page_starts):
    """Return the 1 indexed page number containing character position `pos`.

    page_starts is sorted, so this is a binary search for the last page that
    begins at or before `pos`. bisect_right returns the 0 indexed page plus
    one, which is exactly the 1 indexed page number.
    """
    return bisect_right(page_starts, pos)


def chunk_text(text, page_starts, size=2000, overlap=200):
    """Turn (text, page_starts) from extract_pdf_text into a list of chunks.

    Each chunk is a dict:
        id          position in the returned list, starting at 0
        text        text[start:end]
        start, end  character offsets into `text`, end exclusive
        page_start  1 indexed page holding the first character
        page_end    1 indexed page holding the last character

    Chunks cover the whole text in order and overlap by roughly `overlap`
    characters. The text is first split into pieces of at most
    max(overlap, size // 4) so that packing has fine enough granularity to
    produce that overlap.
    Returns [] for empty text.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")
    if not text:
        return []
    if not page_starts:
        page_starts = [0]

    # Split finer than `size`. Packing can only produce overlap when the
    # pieces are small compared to `overlap`; extracted PDF text often has no
    # blank lines, so splitting at `size` yields page sized pieces and the
    # overlap silently disappears.
    granularity = max(overlap, size // 4)
    spans = split_recursive(text, 0, len(text), granularity)
    chunks = []
    for i, (start, end) in enumerate(pack_greedy(spans, size, overlap)):
        chunks.append({
            "id": i,
            "text": text[start:end],
            "start": start,
            "end": end,
            "page_start": page_of(start, page_starts),
            "page_end": page_of(end - 1, page_starts),  # end is exclusive
        })
    return chunks
