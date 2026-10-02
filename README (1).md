# 🧠 Build Your Own Brain — GDG HWU RAG Workshop Starter

Starter repo for the GDG on Campus Heriot-Watt University Dubai RAG workshop series. Fork this, open the notebook in Colab, and follow along live.

## How to use this (for attendees)

1. Fork this repository to your own GitHub account.
2. Open **`GDG_HWU_RAG_Starter.ipynb`** in Google Colab. Easiest way: go to [colab.research.google.com](https://colab.research.google.com), choose **File → Open notebook → GitHub**, and paste your fork's URL.
3. Have your free Gemini API key ready (get one in advance at https://aistudio.google.com/apikey — see the pre-session email).
4. Run the cells top to bottom during the session.

## What's in this repo

| File | Purpose |
|---|---|
| `GDG_HWU_RAG_Starter.ipynb` | The full workshop notebook — setup, the shared demo build (Part 1), and the "bring your own topic" build (Part 2) |
| `sample_dataset.txt` | A small backup dataset (Burj Khalifa, plain text) in case live Wikipedia fetches are flaky on the day — swap this in via `load_text("txt", "sample_dataset.txt")` |

## What this notebook teaches

- **Universal ingestion**: one `load_text()` function that accepts a Wikipedia article title, a shared Google Doc link, a PDF, a `.txt` file, or pasted text — and always returns clean plain text, with automatic size-checking so every topic stays within the free API tier's limits
- **Chunking**: a hand-rolled fixed-size chunker with overlap, so the mechanics are visible rather than hidden behind a library
- **Embedding**: Google's `gemini-embedding-001` model, called in small rate-limited batches
- **Retrieval**: manual cosine similarity search in plain NumPy — the same core mechanism every vector database uses under the hood

This notebook deliberately stops at **retrieval** — no generation/answering yet. That's Session 2.

## For organizers: picking the size guardrails

`MIN_CHARS` and `MAX_CHARS` in Step 1 of the notebook control how much text any dataset is allowed to contain. **Loading and chunking text costs nothing, regardless of size** — no API call happens until the embedding step, so these caps exist purely to control how many embedding *requests* get fired off live.

- **`MIN_CHARS = 1500`** — below this, there isn't enough material for retrieval to meaningfully demonstrate "search."
- **`MAX_CHARS = 150000`** — roughly a short book / multi-article bundle, ~300 chunks. Embedding is batched at Gemini's real maximum (100 texts/request), so this embeds in ~3 requests — seconds, not minutes.

Rough timing at different sizes (batched at 100 texts/request):

| Text size | Approx. chunks | Approx. API calls | Approx. time |
|---|---|---|---|
| 150,000 chars (current cap) | ~300 | ~3 | Seconds |
| 1,000,000 chars (1MB) | ~2,200 | ~23 | ~1.5–2 min |
| 1GB | ~2,000,000 | ~20,000 | Hours — not live-feasible on any tier |

`embed_texts()` also checkpoints progress to a local pickle file as it runs — if a cell is re-run after an error, it resumes instead of re-embedding (and re-spending API calls on) everything from scratch. This matters more in practice than the raw dataset size does, since live debugging/re-runs are the most likely source of unexpectedly high API usage on the day.

**Before the event:** do a real dry run on the actual venue wifi with a real free-tier key — published rate-limit numbers aren't fully reliable for planning, and an empirical test beats documentation here.

If you need to loosen or tighten the cap for your room size or API quota, `MAX_CHARS` is the one number to change — everything else in the pipeline adapts automatically.

## Before the session (organizer checklist)

- [ ] Send the pre-session email with the Gemini API key walkthrough (students must arrive with a key already created)
- [ ] Send the topic submission form (topic + source type + link/file) so sizes can be sanity-checked in advance
- [ ] Pre-test the Wikipedia fetch and Google Doc export on the day's wifi — if the venue's network is flaky, have students fall back to `"txt"`/`"paste"` with content pasted in advance
- [ ] Confirm every team has an *individual* free Gemini API key — never share one key across a room, it will hit rate limits almost immediately
