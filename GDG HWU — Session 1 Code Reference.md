# GDG HWU — Session 1 Complete Code Reference

**For facilitators — study this before teaching.** This is every line of code in `GDG_HWU_RAG_Starter.ipynb`, in execution order, pulled directly from the notebook so there's zero drift between what you study here and what runs live. 

---

## 0. Install

```python
!pip install -q google-genai numpy wikipedia pypdf requests
print("✅ Packages installed.")
```

Five packages: Gemini's SDK, NumPy for the math and three format-specific helpers (`wikipedia`, `pypdf`, `requests` for Google Docs export).

---

## 1. API key + client

```python
from getpass import getpass

API_KEY = getpass("Paste your Gemini API key here and press Enter: ")

from google import genai
client = genai.Client(api_key=API_KEY)

print("✅ Client ready.")
```

`getpass` hides the key as it's typed and keeps it out of the notebook file itself. `client` is reused for every embedding call from here on — it's the one object that actually talks to Gemini.

---

## 2. The universal loader

This is the piece that makes "any topic" work without needing different code per team.

```python
import re
import requests

# --- Size guardrails ---
# Loading and chunking text is FREE regardless of size -- no API call happens until
# the embedding step below. MAX_CHARS exists only to control how many embedding
# *requests* we fire off live, since that's the one step that touches Gemini's
# rate limits. At ~500-char chunks, 150,000 chars is roughly 300 chunks --
# a real short book / multi-article bundle, not a toy snippet -- and with batched
# embedding (100 chunks/request) that's only ~3 API calls, done in seconds.
# MIN_CHARS is a floor below which there's not really enough material for
# "search" to mean anything.
MIN_CHARS = 1500
MAX_CHARS = 150000


def _extract_gdoc_id(url_or_id):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_id)
    return match.group(1) if match else url_or_id


def _load_wikipedia(title):
    import wikipedia
    wikipedia.set_lang("en")
    page = wikipedia.page(title, auto_suggest=False)
    return page.content


def _load_gdoc(url_or_id):
    doc_id = _extract_gdoc_id(url_or_id)
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    resp = requests.get(export_url)
    resp.raise_for_status()
    return resp.text


def _load_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages_text)
    if not text.strip():
        raise ValueError(
            "No text could be extracted. This is likely a scanned/image PDF, "
            "which this workshop doesn't support — pick a text-based PDF instead."
        )
    return text


def _load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_text(source_type, source):
    """Universal loader: returns clean plain text regardless of source type,
    and automatically enforces a safe size range for the free API tier."""
    loaders = {
        "wikipedia": _load_wikipedia,
        "gdoc": _load_gdoc,
        "pdf": _load_pdf,
        "txt": _load_txt,
        "paste": lambda s: s,
    }
    if source_type not in loaders:
        raise ValueError(f"Unknown source_type '{source_type}'. Choose one of: {list(loaders.keys())}")

    text = loaders[source_type](source).strip()
    char_count = len(text)
    print(f"Loaded {char_count:,} characters.")

    if char_count < MIN_CHARS:
        print(
            f"⚠️  That's quite short (under {MIN_CHARS:,} chars). Retrieval needs a bit more "
            f"material to actually search over — consider a longer article or section."
        )

    if char_count > MAX_CHARS:
        text = text[:MAX_CHARS]
        print(
            f"⚠️  Trimmed from {char_count:,} to {MAX_CHARS:,} characters to stay safely inside "
            f"the free API tier's rate limits. Tip: pick ONE focused article/section, not an "
            f"entire wiki or book."
        )

    return text
```

**Explain:** `load_text(source_type, source)` is called the same way regardless of what the content actually is — a Wikipedia title, a Google Doc link, a file path, or raw pasted text. Everything downstream (chunking, embedding, retrieval) only ever sees a plain string. The size check happens here, before anything touches the API — the warning is printed, not silently swallowed, so students see exactly when and why trimming happened.

```python
demo_text = load_text("wikipedia", "Burj Khalifa")
print(demo_text[:500], "...")
```

---

## 3. Chunking

```python
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


demo_chunks = chunk_text(demo_text)
print(f"Split into {len(demo_chunks)} chunks.")
print("\n--- Example chunk ---\n")
print(demo_chunks[0])
```

**Explain:** `start += chunk_size - overlap` is the one line doing the real work — it means each new chunk starts 50 characters *before* the previous one ended, so a sentence sitting right on a chunk boundary still appears whole in at least one chunk. No API call anywhere in this function — it's why chunking is "free" regardless of document size.

---

## 4. Embedding

```python
import time
import pickle
import os
import numpy as np

MAX_BATCH = 100  # Gemini's actual per-request limit for embed_content


def embed_texts(client, texts, checkpoint_path=None, delay_seconds=1):
    """Embeds a list of texts in batches of up to 100 (Gemini's real max per
    request). If checkpoint_path is given, progress is saved after every batch --
    re-running this after an error resumes instead of starting over."""
    start_index = 0
    all_embeddings = []

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            saved = pickle.load(f)
        if saved["texts_hash"] == hash(tuple(texts)):
            all_embeddings = saved["embeddings"]
            start_index = len(all_embeddings)
            print(f"Resuming from checkpoint: {start_index}/{len(texts)} already embedded.")

    for i in range(start_index, len(texts), MAX_BATCH):
        batch = texts[i:i + MAX_BATCH]
        for attempt in range(3):
            try:
                result = client.models.embed_content(model="gemini-embedding-001", contents=batch)
                all_embeddings.extend([e.values for e in result.embeddings])
                break
            except Exception as e:
                print(f"  Hit an error ({e}), retrying in 5s...")
                time.sleep(5)

        print(f"  Embedded {min(i + MAX_BATCH, len(texts))}/{len(texts)} chunks...")

        if checkpoint_path:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"texts_hash": hash(tuple(texts)), "embeddings": all_embeddings}, f)

        time.sleep(delay_seconds)

    return np.array(all_embeddings)


demo_embeddings = embed_texts(client, demo_chunks, checkpoint_path="demo_checkpoint.pkl")
print(f"\n✅ Done. Embedding matrix shape: {demo_embeddings.shape}")
```

**Explain:** This is the *only* function in the whole pipeline that calls Gemini. Three things worth pointing out live:

- `MAX_BATCH = 100` — Gemini's real per-request ceiling, so a few hundred chunks costs only a handful of API calls, not one-per-chunk.
- The `for attempt in range(3)` retry loop — if a single batch fails (network blip, momentary rate limit), it waits 5 seconds and tries again rather than crashing the whole run.
- The checkpoint block — progress is saved to a local pickle file after every batch. If this cell gets re-run (a student fixes a typo and hits run again), it resumes from where it left off instead of re-embedding from zero and spending API calls twice.

The output shape, e.g. `(312, 768)`, is worth pausing on: 312 chunks, each represented as 768 numbers. That's the embedding matrix, in the flesh.

---

## 5. Retrieval

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def retrieve(client, query, chunks, chunk_embeddings, top_k=3):
    query_result = client.models.embed_content(model="gemini-embedding-001", contents=query)
    query_embedding = np.array(query_result.embeddings[0].values)

    scores = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

**Explain, slowly — this is the payoff moment of the whole session:**

1. `cosine_similarity` measures how aligned two vectors are. The `+ 1e-10` is just a safety net against dividing by zero — not conceptually important, skip it in the explanation unless someone asks.
2. The query gets embedded with the *exact same model* as the chunks — this matters: query and chunks have to live in the same "meaning space" for comparison to make sense.
3. `scores = [cosine_similarity(...) for emb in chunk_embeddings]` — this one line *is* vector search. For every chunk, how similar is it to the question. That's the entire algorithm underneath every vector database, written out in plain sight.
4. `sorted(..., reverse=True)[:top_k]` — highest similarity first, return the top 3.

```python
# Try it! Ask a question about the Burj Khalifa.
query = "How tall is the Burj Khalifa?"

results = retrieve(client, query, demo_chunks, demo_embeddings, top_k=3)

for rank, (chunk, score) in enumerate(results, start=1):
    print(f"#{rank}  (similarity: {score:.3f})")
    print(chunk)
    print("---")
```

```python
# "Try to break it" — a question the article likely doesn't cover well
query = "Try your own question here"

results = retrieve(client, query, demo_chunks, demo_embeddings, top_k=3)
for rank, (chunk, score) in enumerate(results, start=1):
    print(f"#{rank}  (similarity: {score:.3f})")
    print(chunk)
    print("---")
```

**Explain:** low similarity scores across the board on an out-of-scope question are the system honestly saying "I don't have a good answer here" — point this out explicitly as a feature, not a failure.

---

## 6. Part 2 — the same code, their topic

```python
# ---- EDIT THESE TWO LINES ----
MY_SOURCE_TYPE = "wikipedia"        # one of: "wikipedia", "gdoc", "pdf", "txt", "paste"
MY_SOURCE = "Your topic here"       # article title / gdoc link / file path / pasted text
# -------------------------------

my_text = load_text(MY_SOURCE_TYPE, MY_SOURCE)
my_chunks = chunk_text(my_text)
print(f"\n{len(my_chunks)} chunks ready to embed.")
```

```python
my_embeddings = embed_texts(client, my_chunks, checkpoint_path="my_checkpoint.pkl")
print(f"\n✅ Done. Embedding matrix shape: {my_embeddings.shape}")
```

```python
# Ask it something!
my_query = "Ask your bot a question about your own topic"

results = retrieve(client, my_query, my_chunks, my_embeddings, top_k=3)
for rank, (chunk, score) in enumerate(results, start=1):
    print(f"#{rank}  (similarity: {score:.3f})")
    print(chunk)
    print("---")
```

**Explain:** nothing new here — same four functions (`load_text`, `chunk_text`, `embed_texts`, `retrieve`), just called again with a different source. This is the moment to make explicit: *"you now understand every line that just ran."*

---

## Quick function map (for your own mental model while teaching)

| Function | Calls Gemini? | What it does |
| --- | --- | --- |
| `load_text()` | No | Gets clean plain text from any of 5 source types, enforces size limits |
| `chunk_text()` | No | Slices text into overlapping \~500-char pieces |
| `embed_texts()` | **Yes** | Turns a list of chunks into a matrix of vectors, batched, checkpointed |
| `cosine_similarity()` | No | Pure math — how aligned two vectors are |
| `retrieve()` | **Yes** (once, for the query only) | Embeds the question, scores it against every chunk, returns the top matches |