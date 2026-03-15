"""
Phase 1 — Build once (offline): scrape 15 sections + subsections, chunk, embed, store in ChromaDB.

Run this script once (or use POST /scrape-policies). Takes ~2–3 minutes. No scraping at query time.
  python -m backend.scripts.build_policy_index
"""
import time

from backend.tools.policy_index import clear as clear_policy_index, upsert_chunks
from backend.tools.policy_scraper import scrape_all


def run_build():
    """Clear index, scrape all sections/subsections, upsert into ChromaDB. Returns (num_chunks, elapsed_seconds)."""
    start = time.perf_counter()
    clear_policy_index()
    chunks = scrape_all()
    n = upsert_chunks(chunks)
    elapsed = time.perf_counter() - start
    return n, elapsed


if __name__ == "__main__":
    print("Building policy index (Phase 1)...")
    n, elapsed = run_build()
    print(f"Built in {elapsed:.1f}s; {n} chunks indexed.")
    print("Run this script once (or POST /scrape-policies); takes ~2–3 min.")
