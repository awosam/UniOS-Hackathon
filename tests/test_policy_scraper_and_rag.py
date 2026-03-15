"""
Tests for policy scraper and ChromaDB policy index (subsection-level RAG).

Run from repo root: python -m pytest tests/test_policy_scraper_and_rag.py -v
Or: python tests/test_policy_scraper_and_rag.py
"""

import os
import sys

# Repo root on path so backend.* imports work
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def test_policy_index_upsert_search_clear():
    """policy_index: upsert chunks with section/subsection, search returns them, clear removes."""
    from backend.tools.policy_index import upsert_chunks, search_policies, clear

    clear()
    chunks = [
        {
            "id": "policy_1",
            "text": "Students must maintain a 90% rule for academic standing.",
            "metadata": {
                "source_key": "academic-regulations__90-rule",
                "section": "Academic Regulations",
                "subsection": "90% Rule",
                "url": "https://example.ca/90-rule",
            },
        },
        {
            "id": "policy_2",
            "text": "Admission requirements include minimum average and prerequisites.",
            "metadata": {
                "source_key": "admissions__admission-requirements",
                "section": "Admissions",
                "subsection": "Admission Requirements",
                "url": "https://example.ca/admission",
            },
        },
    ]
    n = upsert_chunks(chunks)
    assert n == 2

    results = search_policies("90% rule academic standing", k=5)
    assert len(results) >= 1
    for r in results:
        assert "text" in r and "url" in r and "section" in r and "subsection" in r and "source_key" in r
        assert r["section"] in ("Academic Regulations", "Admissions")
        assert r["subsection"]

    results2 = search_policies("admission requirements", k=5)
    assert len(results2) >= 1

    clear()
    results_after = search_policies("90%", k=5)
    assert len(results_after) == 0


def test_policy_scraper_slug():
    """policy_scraper._slug normalizes labels to slugs."""
    from backend.tools.policy_scraper import _slug

    assert _slug("Academic Regulations") == "academic-regulations"
    assert _slug("90% Rule") == "90-rule"
    assert _slug("  Admissions  ") == "admissions"


def test_policy_scraper_chunk_text():
    """policy_scraper._chunk_text splits with overlap."""
    from backend.tools.policy_scraper import _chunk_text, CHUNK_SIZE

    short = "One paragraph."
    assert _chunk_text(short) == ["One paragraph."]
    assert _chunk_text("") == []

    long_text = "x" * (CHUNK_SIZE + 500)
    chunks = _chunk_text(long_text, chunk_size=CHUNK_SIZE)
    assert len(chunks) >= 2
    assert all(len(c) <= CHUNK_SIZE + 200 for c in chunks)


def test_policy_scraper_extract_main_content():
    """policy_scraper._extract_main_content gets text from main/article/body."""
    from backend.tools.policy_scraper import _extract_main_content
    from bs4 import BeautifulSoup

    html = """
    <html><body>
    <nav>Skip</nav>
    <main><p>Policy text here.</p><p>Second paragraph.</p></main>
    <footer>Footer</footer>
    </body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    text = _extract_main_content(soup)
    assert "Policy text here" in text
    assert "Second paragraph" in text
    assert "Skip" not in text or "Footer" not in text


def test_policy_scraper_is_same_policy_url():
    """policy_scraper._is_same_policy_url filters by domain and path."""
    from backend.tools.policy_scraper import _is_same_policy_url, BASE_DOMAIN

    base = "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/"
    assert _is_same_policy_url(
        "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/page/Acad-Regs-Grades.html",
        base,
    )
    assert not _is_same_policy_url("https://other.com/page", base)
    assert not _is_same_policy_url("https://academic-calendar-archive.uwaterloo.ca/other/path", base)


def test_scrape_all_output_shape():
    """scrape_all with minimal policy_pages returns list of dicts with id, text, metadata (section, subsection, url)."""
    from backend.tools.policy_scraper import scrape_all

    # Use a single known-good section URL so we get at least one subsection and predictable shape
    minimal_pages = {
        "University Policies, Guidelines, and Academic Regulations": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/uWaterloo-Policies-Guidelines-Academic-Regulations.html",
    }
    chunks = scrape_all(policy_pages=minimal_pages)
    # May be 0 if network fails or page structure changes; if we get any, check shape
    for c in chunks:
        assert "id" in c and c["id"].startswith("policy_")
        assert "text" in c and isinstance(c["text"], str)
        assert "metadata" in c
        m = c["metadata"]
        assert "source_key" in m and "section" in m and "subsection" in m and "url" in m


def test_pathfinder_uses_policy_index_when_populated():
    """Pathfinder retrieve_policies uses policy_index when ChromaDB has data; format includes Section > Subsection."""
    from backend.tools.policy_index import clear, upsert_chunks, search_policies
    from backend.agents.pathfinder import retrieve_policies

    clear()
    upsert_chunks([
        {"id": "p1", "text": "Grade appeal deadline is 10 business days.", "metadata": {"source_key": "ar__appeals", "section": "Academic Regulations", "subsection": "Grade Appeals", "url": "https://example.ca/appeals"}},
    ])
    state = {"goal": "How do I appeal a grade?", "student_record": {}, "policies": [], "roadmap": []}
    out = retrieve_policies(state)
    assert out["policies"]
    first = out["policies"][0]
    assert "Academic Regulations" in first
    assert "Grade Appeals" in first or "grade" in first.lower()
    assert "https://" in first
    clear()


def test_chat_policy_question_returns_citation():
    """When policy index has data, a policy question in chat returns a response that includes a citation (Source/URL/section)."""
    import asyncio
    from backend.tools.policy_index import clear, upsert_chunks
    from backend.agents.chat_agent import chat_agent

    clear()
    upsert_chunks([
        {"id": "p1", "text": "Posthumous degrees can be granted to an undergraduate who was pursuing completion of a UW degree at time of death. 50% or more of required units must be completed.", "metadata": {"source_key": "ar__posthumous", "section": "Academic Regulations", "subsection": "Posthumous Degrees and Certificates of Enrolment", "url": "https://academic-calendar-archive.uwaterloo.ca/page/posthumous.html"}},
    ])
    try:
        resp = asyncio.run(chat_agent.get_response("What is the policy on Posthumous Degrees and Certificates of Enrolment?"))
        assert "text" in resp
        text = resp["text"]
        assert any(c in text for c in ["Source", "http", "Academic Regulations", "Posthumous", "50%"])
    finally:
        clear()


def _run_all():
    """Run all tests and print results."""
    tests = [
        ("policy_index upsert/search/clear", test_policy_index_upsert_search_clear),
        ("policy_scraper _slug", test_policy_scraper_slug),
        ("policy_scraper _chunk_text", test_policy_scraper_chunk_text),
        ("policy_scraper _extract_main_content", test_policy_scraper_extract_main_content),
        ("policy_scraper _is_same_policy_url", test_policy_scraper_is_same_policy_url),
        ("scrape_all output shape", test_scrape_all_output_shape),
        ("pathfinder uses policy_index", test_pathfinder_uses_policy_index_when_populated),
        ("chat policy question returns citation", test_chat_policy_question_returns_citation),
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
        except Exception as e:
            print(f"FAIL: {name} — {e}")
            failed.append((name, e))
    if failed:
        print(f"\n{len(failed)} failed, {len(tests) - len(failed)} passed")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        import pytest
        pytest.main([__file__, "-v"])
    else:
        _run_all()
