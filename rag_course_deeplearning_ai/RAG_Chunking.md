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

## Why Chunk — Three Pillars

The course frames the case for chunking around three motivations:

1. **Token limits** — embedding models truncate beyond 512 tokens; LLMs have finite context windows.
2. **Improved relevance** — small chunks produce sharper, more specific embeddings that match queries better than blurred whole-document averages.
3. **LLM only sent relevant context** — when retrieved chunks are tight, the LLM's attention isn't diluted across irrelevant material.

### Indexing without chunking — failure mode

If you embed each of 1,000 books as a single vector, you get 1,000 "averaged" representations. A query about a specific recipe in a cookbook returns the *cookbook* as the closest match — but then the LLM gets thousands of irrelevant tokens jammed into its context.

---

## The Goldilocks Framing

| Chunk size | Problem |
|------------|---------|
| **Too large** (chapter) | Each chunk contains many topics → less specific embeddings → fills LLM context window |
| **Too small** (word/phrase) | Loses surrounding context → reduces retrieval relevance |
| **Just right** | Topically coherent, context-rich, but specific |

There is no universal sweet spot — the right size depends on query specificity, document structure, and retrieval method.

---

## Strategy 5 — Recursive Character Splitting

Split at a hierarchy of separators in order, falling back to the next when chunks are still too large. Respects natural language structure better than fixed-size word counts.

```python
# Pseudocode for recursive splitting
SEPARATORS = ["\n\n", "\n", ". ", " "]  # try paragraph → line → sentence → word

def recursive_split(text: str, max_size: int, separators=SEPARATORS) -> list[str]:
    if len(text) <= max_size or not separators:
        return [text]
    sep, *rest = separators
    chunks = []
    for part in text.split(sep):
        if len(part) <= max_size:
            chunks.append(part)
        else:
            chunks.extend(recursive_split(part, max_size, rest))
    return chunks
```

**Concrete example** — a 3-sentence Taylor Swift / Eras Tour passage:
- Try `\n\n` → still too big
- Fall back to `. ` → splits into 3 clean sentences

This is the default chunker in many production pipelines (e.g., LangChain's `RecursiveCharacterTextSplitter`).

### Language-specific variants

- **HTML** — split on `<p>`, `<h1>`, `<div>` boundaries.
- **Python source** — split at `def` and `class` boundaries.
- **Markdown** — split at `#`, `##`, `###` headers (already implemented in many libraries).

---

## Strategy 6 — Semantic Chunking

Use cosine similarity between consecutive sentences to detect topic boundaries. If similarity drops below a threshold, start a new chunk.

```python
def semantic_chunk(text: str, model, threshold: float = 0.7) -> list[str]:
    sentences = text.split(". ")
    embeddings = model.encode(sentences)
    
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunks.append(". ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(". ".join(current))
    return chunks
```

**Example:** *"Canada is known for its Maple syrup. The country has beautiful mountains. Canadian landscapes are stunning."* — high inter-sentence similarity → stays as one chunk. Useful for prose where topic boundaries aren't structurally marked.

Reference: Kamradt, G. (2023). *5 Levels of Text Splitting*.

---

## Strategy 7 — LLM-Based Chunking

Prompt an LLM to split a document into semantically coherent chunks. Most expensive option but handles edge cases (e.g., conversational transcripts, code with embedded comments) that mechanical splitters mangle.

```python
PROMPT = """Split the following text into coherent chunks. Each chunk should cover one topic.
Return chunks separated by `<CHUNK>` markers. Keep related concepts together.

Text: {text}"""
```

Cost: one LLM call per document at indexing time. Acceptable for one-time corpus builds; prohibitive for streaming ingest.

---

## Strategy 8 — Context-Aware Chunking

After splitting, ask an LLM to add a brief context label to each chunk so retrieval has more signal than the chunk text alone.

```
Chunk 1: "We launched the app on March 5th."
+ context: "App launch announcement (overview section)"

Chunk 2: "Engagement was 3× our forecast."
+ context: "Q1 metrics — launch results"

Chunk 3: "Lessons: ship faster, test on iPad earlier."
+ context: "Postmortem learnings"
```

Each chunk now embeds both its raw text *and* its context label. Searches that match the topic but not the surface words still find the chunk.

**Tradeoff:** another LLM call per chunk during indexing. The course flags this as **the highest-leverage chunking improvement to try first.**

---

## Decision Recommendations (from the course)

| Strategy | When to choose | Cost |
|----------|---------------|------|
| **Fixed-width** | Default; quick baseline | Free |
| **Fixed-width with overlap** | Default for prose with no structure | Free |
| **Recursive character split** | Documents with hierarchy (paragraphs, sentences) | Free |
| **Variable on structure** | Markdown, AsciiDoc, structured docs | Free |
| **Mixed (variable + min-size merge)** | Real-world structured docs | Free |
| **Semantic chunking** | Prose without structure but with topic boundaries | Embedding model calls |
| **LLM-based chunking** | Conversational or unusual formats | LLM calls per doc |
| **Context-aware chunking** | High-value corpora where retrieval quality matters most | LLM calls per chunk |

The course's takeaway: *Fixed-width and recursive character splitting are "good defaults"; context-aware chunking is the "good first improvement to explore" once the baseline works.*

---

## See Also

- [[RAG_Embeddings]] — why the 512-token limit matters
- [[RAG_Weaviate]] — loading chunks into a vector database
- [[RAG_Retrieval_Methods]] — how chunk size affects retrieval quality
- [[RAG_Advanced_Retrieval]] — ColBERT and per-token retrieval as an alternative to chunking
