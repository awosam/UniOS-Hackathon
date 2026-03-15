"""
policy_scraper.py — Scrape UW undergrad calendar policy sections at subsection level for RAG.

Scope: 15 policy section entry-point URLs only (catalog#/policies). For each section we
discover subsection links from the section page, fetch each subsection, extract main
content, chunk text, and tag each chunk with section, subsection, and url for citations.

Option A: Subsection discovery from section pages (no hardcoded subsection URLs).
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

# Chunk size aligned with policy_decoder (1200 chars) for consistent RAG context
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100

# Base for same-domain and same-path filtering (undergrad calendar archive)
BASE_DOMAIN = "academic-calendar-archive.uwaterloo.ca"
BASE_NETLOC = f"https://{BASE_DOMAIN}"

# 15 policy section entry-point URLs (keys = section labels, values = full section page URLs).
# Source: UW undergrad calendar policy tree (catalog#/policies). Update paths if catalog structure changes.
POLICY_PAGES: Dict[str, str] = {
    "University Policies, Guidelines, and Academic Regulations": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/uWaterloo-Policies-Guidelines-Academic-Regulations.html",
    "Academic Regulations": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/uWaterloo-Policies-Guidelines-Academic-Regulations.html",
    "Admissions": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/Admissions.html",
    "Awards and Financial Aid": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/Awards-Financial-Aid.html",
    "Co-operative Education and Career Action": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/Co-operative-Education-Career-Action.html",
    "Fees and Charges": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/Fees-Charges.html",
    "Graduation": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/Graduation.html",
    "Student Awards": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/Student-Awards.html",
    "Services and Resources": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/Services-Resources.html",
    "Faculties, Schools, and Colleges": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/Faculties-Schools-Colleges.html",
    "Regulations and Requirements": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/Regulations-Requirements.html",
    "Academic Integrity": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/Academic-Integrity.html",
    "Calendar of Events and Deadlines": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/Calendar-Events-Deadlines.html",
    "Policies and Guidelines": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/Policies-Guidelines.html",
    "About the Calendar": "https://academic-calendar-archive.uwaterloo.ca/undergraduate-studies/2023-2024/group/About-Calendar.html",
}


def _slug(s: str) -> str:
    """Normalize a label to a short slug for source_key."""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _is_same_policy_url(url: str, base: str) -> bool:
    """True if url is same domain and under undergrad calendar path (policy tree)."""
    try:
        p = urlparse(url)
        return (
            BASE_DOMAIN in (p.netloc or "")
            and (p.path or "").strip("/").startswith("undergraduate-studies/")
        )
    except Exception:
        return False


def _extract_main_content(soup: BeautifulSoup) -> str:
    """Extract main text from legacy content div or <main> fallback."""
    # Prefer main content area; avoid nav/footer
    main = soup.find("main") or soup.find("article") or soup.find(class_=re.compile(r"content|main|body", re.I))
    if main:
        for tag in main.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()
        return re.sub(r"\s+", " ", main.get_text(separator=" ", strip=True))
    # Fallback: body without nav/footer
    body = soup.find("body")
    if body:
        for tag in body.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return re.sub(r"\s+", " ", body.get_text(separator=" ", strip=True))
    return ""


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)
    return chunks


def _discover_subsection_links(
    section_url: str,
    section_label: str,
    soup: BeautifulSoup,
    seen_urls: set,
) -> List[Tuple[str, str]]:
    """
    Find subsection links in the section page (main content area, same domain/policy path).
    Returns list of (subsection_url, subsection_label). Dedupes by URL; first section owns it.
    """
    base = section_url.rsplit("/", 1)[0] + "/"
    out: List[Tuple[str, str]] = []
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if not main:
        return out
    for a in main.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#"):
            continue
        full_url = urljoin(section_url, href)
        if not _is_same_policy_url(full_url, base):
            continue
        if full_url in seen_urls:
            continue
        label = (a.get_text() or "").strip() or None
        seen_urls.add(full_url)
        out.append((full_url, label or "Overview"))
    return out


def _fetch_page(client: httpx.Client, url: str) -> Optional[BeautifulSoup]:
    """GET url and return BeautifulSoup or None on failure."""
    try:
        r = client.get(url, follow_redirects=True, timeout=30)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception:
        return None


def _get_page_title(soup: BeautifulSoup) -> str:
    """Title from <title> or first h1."""
    t = soup.find("title")
    if t and t.get_text(strip=True):
        return t.get_text(strip=True)
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    return "Overview"


def scrape_all(
    policy_pages: Optional[Dict[str, str]] = None,
    chunk_size: int = CHUNK_SIZE,
) -> List[Dict[str, Any]]:
    """
    Scrape 15 section entry pages, discover subsections, fetch and chunk each subsection.
    Returns list of dicts: {"id": str, "text": str, "metadata": {"source_key", "section", "subsection", "url"}}.
    """
    pages = policy_pages or POLICY_PAGES
    seen_urls: set = set()
    results: List[Dict[str, Any]] = []
    chunk_id = 0

    with httpx.Client() as client:
        for section_label, section_url in pages.items():
            soup = _fetch_page(client, section_url)
            if not soup:
                # No subsections: treat section page as one subsection
                subsections: List[Tuple[str, Optional[str]]] = [(section_url, "Overview")]
            else:
                subs = _discover_subsection_links(section_url, section_label, soup, seen_urls)
                subsections = [(url, label) for url, label in subs]

            if not subsections:
                # No links found: scrape section page as single subsection
                subsections = [(section_url, _get_page_title(soup) if soup else "Overview")]

            for sub_url, sub_label in subsections:
                sub_soup = _fetch_page(client, sub_url)
                raw = _extract_main_content(sub_soup) if sub_soup else ""
                if not raw:
                    continue
                sub_label_resolved = sub_label if sub_label else (_get_page_title(sub_soup) if sub_soup else "Overview")
                source_key = f"{_slug(section_label)}__{_slug(sub_label_resolved)}"
                for chunk_text in _chunk_text(raw, chunk_size=chunk_size):
                    chunk_id += 1
                    doc_id = f"policy_{chunk_id}"
                    results.append({
                        "id": doc_id,
                        "text": chunk_text,
                        "metadata": {
                            "source_key": source_key,
                            "section": section_label,
                            "subsection": sub_label_resolved,
                            "url": sub_url,
                        },
                    })

    return results
