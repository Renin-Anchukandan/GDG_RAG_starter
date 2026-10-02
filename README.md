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

`MIN_CHARS` and `MAX_CHARS` in Step 1 of the notebook control how much text any dataset is allowed to contain:

- **`MIN_CHARS = 1500`** — below this, there isn't enough material for retrieval to meaningfully demonstrate "search."
- **`MAX_CHARS = 15000`** — roughly 6–8 pages / one focused Wikipedia article. This keeps every team at about 25–40 chunks, which keeps embedding calls (and therefore free-tier rate-limit risk) manageable across a room of teams all building at once.

If you need to loosen or tighten these for your specific room size or API quota, they're the two numbers to change — everything else in the pipeline adapts automatically.

## Before the session (organizer checklist)

- [ ] Send the pre-session email with the Gemini API key walkthrough (students must arrive with a key already created)
- [ ] Send the topic submission form (topic + source type + link/file) so sizes can be sanity-checked in advance
- [ ] Pre-test the Wikipedia fetch and Google Doc export on the day's wifi — if the venue's network is flaky, have students fall back to `"txt"`/`"paste"` with content pasted in advance
- [ ] Confirm every team has an *individual* free Gemini API key — never share one key across a room, it will hit rate limits almost immediately
