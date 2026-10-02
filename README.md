# GDG HWU RAG Workshop Starter

Starter repo for the GDG on Campus Heriot-Watt University Dubai RAG workshop series. 
Fork this, open the notebook in Colab and follow along live.

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
