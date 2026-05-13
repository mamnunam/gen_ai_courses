# Chunking

## Why Chunk?

Embedding models have a hard token limit (512 tokens for `bge-base-en-v1.5`). Text beyond that limit is **silently truncated** — the embedding is identical to a shorter version of the same text. See [[RAG_Embeddings]] for the proof.

Beyond the technical limit, chunking also improves retrieval quality:

- A single embedding for a 5,000-word article averages everything — the vector becomes vague and hard to match precisely.
- Smaller chunks produce sharper, more focused embeddings that match specific queries better.

### The Fundamental Tradeoff

| Chunk Size | Retrieval Precision | LLM Context Quality |
|------------|--------------------|--------------------|
| Very small (≤25 words) | High — pinpoints the exact sentence | Low — little surrounding context |
| Medium (~100 words) | Good | Good |
| Large (paragraph+) | Lower — may drift semantically | High — richer, coherent context |
| Too large (>512 tokens) | Poor — truncated | —|

**General rule:** Specific factual queries → smaller chunks. Broad conceptual queries → larger chunks.

---

## Strategy 1 — Fixed-Size Chunking

Split text into equal word-count chunks. Simple and predictable.

```python
from typing import List

def get_chunks_fixed_size(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of exactly chunk_size words (last chunk may be shorter)."""
    words  = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

chunks = get_chunks_fixed_size(article_text, chunk_size=100)
print(f"{len(chunks)} chunks")
```

**Problem:** Important sentences can be cut in half at chunk boundaries.

---

## Strategy 2 — Fixed-Size with Overlap

Add a sliding window that includes the tail of the previous chunk. This prevents information from being lost at boundaries.

```
|← chunk_size →|
[====prev chunk====][=========current chunk=========]
                |← overlap →|
```

```python
def get_chunks_fixed_size_with_overlap(
    text: str,
    chunk_size: int,
    overlap_fraction: float
) -> List[str]:
    """Fixed-size chunks where each chunk repeats overlap_fraction of the previous one."""
    words   = text.split()
    overlap = int(chunk_size * overlap_fraction)
    chunks  = []

    for i in range(0, len(words), chunk_size):
        # Start from `overlap` words before position i (except for the first chunk)
        chunk_words = words[max(i - overlap, 0) : i + chunk_size]
        chunks.append(" ".join(chunk_words))

    return chunks

# 100-word chunks with 20% overlap → adjacent chunks share ~20 words
chunks = get_chunks_fixed_size_with_overlap(text, chunk_size=100, overlap_fraction=0.2)
```

**Typical overlap:** 15–25%. Too little → boundary artifacts. Too much → redundant embeddings and wasted space.

---

## Strategy 3 — Variable-Size: Paragraph Splits

Split on natural language boundaries rather than enforcing a fixed size. Preserves semantic integrity of paragraphs.

```python
# Split on blank lines (Markdown / plain text paragraph breaks)
def get_chunks_by_paragraph(text: str) -> List[str]:
    return [p for p in text.split("\n\n") if p.strip()]

# Split on section markers (e.g., Asciidoc headers "== Section Title")
def get_chunks_by_section(text: str) -> List[str]:
    return [s for s in text.split("\n==") if s.strip()]

# Split on Markdown headers
def get_chunks_by_markdown_header(text: str) -> List[str]:
    import re
    return re.split(r'\n#{1,3} ', text)
```

**Problem:** Short fragments (like lone headings) become their own chunks — semantically useless and pollute the index.

---

## Strategy 4 — Mixed Strategy (Variable + Minimum Size)

Split on structural markers, then merge any chunk that's too small with the next one.

```python
def mixed_chunking(text: str, separator: str = "\n==", min_words: int = 25) -> List[str]:
    """
    Split on a structural marker, then merge chunks that are too short
    into the next chunk. Prevents lone headings becoming their own chunks.
    """
    raw_chunks  = text.split(separator)
    new_chunks  = []
    buffer      = ""

    for chunk in raw_chunks:
        candidate = buffer + chunk
        if len(candidate.split()) < min_words:
            buffer = candidate          # too short — carry forward
        else:
            new_chunks.append(candidate)
            buffer = ""

    if buffer:
        new_chunks.append(buffer)       # flush any remaining text

    return new_chunks
```

---

## Comparing Strategies in Practice

On the same query (`"history of git"`):

| Strategy | Result |
|----------|--------|
| `fixed_size_25` | Finds the exact sentence mentioning Git's origin — but lacks context |
| `fixed_size_100` | Returns a readable paragraph about Git history |
| `para_chunks` | Returns full section — comprehensive but potentially drifts |
| `para_chunks_min_25` | **Best general result** — coherent sections without lone headings |

On a specific query (`"how to add the url of a remote repository"`):
- `fixed_size_25` wins — directly retrieves the `git remote add <shortname> <url>` command.
- Larger chunks bury the specific command in a wall of text.

---

## Chunk Objects: Adding Metadata

When loading chunks into a vector database, always attach metadata alongside the text. This enables metadata filtering later.

```python
def build_chunk_objects(source_obj: dict, chunks: List[str]) -> List[dict]:
    """Attach source metadata to each chunk."""
    return [
        {
            "chapter_title": source_obj["chapter_title"],
            "filename":      source_obj["filename"],
            "chunk":         chunk_text,
            "chunk_index":   i,
        }
        for i, chunk_text in enumerate(chunks)
    ]

# Build chunks from multiple documents with different strategies
chunk_sets = {}
for doc in documents:
    text = doc["body"]
    for strategy_name, chunks in [
        ("fixed_25",      get_chunks_fixed_size_with_overlap(text, 25,  0.2)),
        ("fixed_100",     get_chunks_fixed_size_with_overlap(text, 100, 0.2)),
        ("para",          get_chunks_by_paragraph(text)),
        ("para_min_25",   mixed_chunking(text)),
    ]:
        chunk_objs = build_chunk_objects(doc, chunks)
        chunk_sets.setdefault(strategy_name, []).extend(chunk_objs)
```

---

## BBC News Dataset: Pre-Chunked Schema

In Modules 3–5, the BBC News articles are already chunked before being loaded into Weaviate. Each chunk is stored as a separate object:

```python
{
    "title":           "Taylor Swift thanks fans after Wembley record",
    "pubDate":         "2024-08-21T03:02:08+00:00",
    "guid":            "cr5nr3n6epvo",
    "link":            "https://www.bbc.com/news/articles/...",
    "description":     "The star is joined by Florence + The Machine...",
    "article_content": "Taylor Swift has finished the European leg...",  # full article
    "chunk":           "size crowd at all. At an earlier show...",       # this chunk's text
    "chunk_index":     10                                                # position in article
}
```

The `chunk` field is what gets embedded and stored as a 768-dim vector.

---

## Decision Guide

```
Is the text short enough to fit in 512 tokens?
  └─ YES → embed directly, no chunking needed
  └─ NO  → must chunk before embedding

Is the query very specific (looking for a single fact or command)?
  └─ YES → small chunks (25–50 words) — maximizes precision
  └─ NO  → larger chunks (100+ words) or paragraphs

Does context continuity matter (code, step-by-step instructions)?
  └─ YES → use overlap (20–25% of chunk size)
  └─ NO  → fixed-size without overlap is fine

Is the document structured (headers, sections, chapters)?
  └─ YES → variable-size on structural markers + min-size merge
  └─ NO  → fixed-size with overlap as a safe default

Are chunks too small after splitting on structure?
  └─ YES → mixed strategy: merge tiny chunks with the next one
```

---

## See Also

- [[RAG_Embeddings]] — why the 512-token limit matters
- [[RAG_Weaviate]] — loading chunks into a vector database
- [[RAG_Retrieval_Methods]] — how chunk size affects retrieval quality
