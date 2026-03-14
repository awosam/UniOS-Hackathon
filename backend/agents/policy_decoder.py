"""
policy_decoder.py — UW policy PDF ingestion and AI-powered question answering.

WHY THIS EXISTS:
  University policy is vast and hard to navigate. Students often don't know:
  - Which form to submit for a grade appeal
  - What the exact deadline is for dropping a course
  - Whether their situation qualifies for special consideration

  PolicyDecoder ingests official UW policy PDFs and lets students ask natural-
  language questions. It combines two techniques:
    1. Keyword retrieval: finds the most relevant policy text chunks
    2. AI synthesis:     Gemini reads those chunks and writes a clear answer

  This pattern is called RAG — Retrieval-Augmented Generation. It's more accurate
  than asking the AI directly (which relies on potentially outdated training data)
  because the AI is grounded in the actual current policy document.

HOW TO ADD A POLICY DOCUMENT:
  Upload a PDF via the /ingest-policy endpoint, or call directly:
    policy_decoder.ingest_pdf("path/to/uw-academic-calendar.pdf", "UW Calendar 2026")

STORAGE:
  Extracted text chunks are stored in backend/data/policy_chunks.json.
  This file persists between server restarts so you don't re-ingest every time.
"""

import json          # For reading/writing chunk storage to disk
import os            # For file path operations and directory creation
import re            # For cleaning whitespace from extracted PDF text
from concurrent.futures import ThreadPoolExecutor  # Not currently used; reserved for async extension
from dataclasses import dataclass    # Lightweight data container for policy chunks
from typing import List, Optional    # Type hints

import vertexai
from vertexai.generative_models import GenerativeModel
from pypdf import PdfReader          # Third-party library for extracting text from PDFs

from backend.vertex import VERTEX_PROJECT, VERTEX_LOCATION  # Shared GCP config

# Initialize Vertex AI at module load
vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)

# Reserved for future async extension — not actively used currently
_executor = ThreadPoolExecutor(max_workers=4)

# Gemini model for synthesizing answers from retrieved policy chunks
_model = GenerativeModel("gemini-2.5-flash")


@dataclass
class PolicyChunk:
    """
    Represents one page-sized piece of text from a policy PDF.

    WHY CHUNKS AND NOT THE WHOLE DOCUMENT:
      Sending an entire 200-page academic calendar to Gemini would be:
      1. Extremely slow (large context window = longer generation time)
      2. Expensive (billed per token)
      3. Less accurate (Gemini focuses better on a targeted excerpt)

      Instead we split each PDF page into ~1200-character chunks, then
      retrieve only the 5 most relevant chunks for each question.

    Attributes:
        source: Human-readable name of the source document (e.g. "UW Calendar 2026")
        page:   1-indexed page number in the original PDF (for citation)
        text:   The extracted text content of this chunk
    """
    source: str
    page: int
    text: str


class PolicyDecoder:
    """
    Ingests UW policy PDFs and answers policy questions using RAG.

    RETRIEVAL STRATEGY — KEYWORD SCORING:
      We use a simple word-counting approach: for each query word, count how many
      times it appears in each chunk. The chunks with the highest total counts are
      returned as the most relevant.

      WHY NOT VECTOR EMBEDDINGS:
        Embedding-based retrieval is more semantically accurate but requires:
        - A vector database (Pinecone, ChromaDB, etc.)
        - An embedding model API call for every chunk at ingest time
        - An embedding call for every question at query time
        For a hackathon with policy documents that use precise legal language,
        keyword matching is 80% as good with 0% of the infrastructure complexity.
    """

    def __init__(self):
        # Absolute path to the JSON file that persists chunks across server restarts
        self._store_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "policy_chunks.json")
        )
        # Load any previously ingested chunks immediately
        self._chunks: List[PolicyChunk] = self._load_chunks()

    def query_with_url(self, query: str, url: str = None) -> str:
        """
        Answers a policy question using Gemini's training knowledge, optionally
        mentioning a URL for the student to verify the answer themselves.

        WHY THIS FALLBACK EXISTS:
          If no PDFs have been ingested yet, we can still answer policy questions
          using Gemini's general knowledge of UW policies (from training data).
          The URL gives students a reference to verify the answer.

        Args:
            query: The student's policy question in natural language.
            url:   Optional UW policy page URL to cite in the response.
        """
        url_context = f"Relevant policy reference: {url}\n" if url else ""
        prompt = (
            "You are Uni-OS, the University of Waterloo academic companion.\n"
            f"{url_context}"
            f"Student policy question: {query}\n\n"
            "Answer accurately using your knowledge of UW academic policies. "
            "Cite the relevant section of the UW academic calendar or registrar guidelines. "
            "If unsure about a specific detail, say so and direct the student to uwaterloo.ca."
        )
        return _model.generate_content(prompt).text

    def query_with_context(self, query: str) -> str:
        """
        Retrieves the most relevant stored policy chunks, then uses Gemini to
        synthesize a clear, cited answer from those excerpts.

        This is the primary RAG method — more accurate than query_with_url because
        it grounds the AI in actual document text rather than training data memory.

        Falls back to query_with_url if no chunks have been ingested yet.
        """
        chunks = self.query_policies(query)

        if not chunks:
            # No ingested documents — fall back to AI's general knowledge
            return self.query_with_url(query)

        # Format the retrieved chunks as labeled context for the AI
        context = "\n\n".join(
            f"[Source: {c.source}, page {c.page}]\n{c.text}"
            for c in chunks
        )

        prompt = (
            "You are Uni-OS, the University of Waterloo academic companion.\n"
            f"Student question: {query}\n\n"
            "Relevant policy excerpts from UW documents:\n"
            f"{context}\n\n"
            "Answer the question based only on these excerpts. "
            "Quote specific page numbers where possible. "
            "If the excerpts don't cover the question, say so clearly."
        )
        return _model.generate_content(prompt).text

    def ingest_pdf(self, file_path: str, source_name: Optional[str] = None) -> int:
        """
        Reads a PDF, splits it into chunks, and saves them for future retrieval.

        CHUNKING STRATEGY:
          Each page's text is extracted, cleaned, then split into 1200-character
          segments. 1200 chars ≈ 200-250 words — enough context for Gemini to
          give a specific answer, small enough that many chunks fit in one prompt.

        Args:
            file_path:   Absolute or relative path to the PDF file.
            source_name: Human-readable name for citations (defaults to filename).

        Returns:
            Number of new chunks created (useful for logging/confirmation).
        """
        reader = PdfReader(file_path)  # pypdf opens and parses the PDF
        src = source_name or os.path.basename(file_path)  # Default to filename if no name given
        chunks: List[PolicyChunk] = []

        for page_idx, page in enumerate(reader.pages):
            # Extract raw text from the page (pypdf tries to preserve reading order)
            text = page.extract_text() or ""

            # Collapse multiple whitespace/newlines into single spaces for cleaner text
            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                continue  # Skip pages with no extractable text (e.g. image-only pages)

            # Split the page text into 1200-character chunks with overlap
            # (no overlap here — adjacent chunks on the same page are retrieved together)
            for i in range(0, len(text), 1200):
                part = text[i: i + 1200].strip()
                if part:  # Skip empty segments at the end
                    chunks.append(PolicyChunk(source=src, page=page_idx + 1, text=part))

        # Add new chunks to the existing store (don't replace — accumulate across PDFs)
        self._chunks.extend(chunks)
        self._persist_chunks()  # Save to disk immediately
        return len(chunks)

    def query_policies(self, query: str, k: int = 5) -> List[PolicyChunk]:
        """
        Returns the top-k most relevant chunks for the given query.

        SCORING ALGORITHM — TERM FREQUENCY:
          For each word in the query (length ≥ 3 to skip stopwords like "is", "at"),
          count how many times it appears in each chunk. Sum the counts — chunks with
          higher sums are more relevant. Return the top k.

        Args:
            query: The search terms or question to match against stored chunks.
            k:     Maximum number of chunks to return. 5 is enough context for most
                   questions while staying within reasonable prompt lengths.
        """
        # Extract meaningful query terms — skip short words (stopwords like "is", "at")
        terms = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) >= 3]

        if not terms or not self._chunks:
            return []  # Nothing to search — return empty

        # Score each chunk by total term frequency across all query terms
        ranked = sorted(
            self._chunks,
            key=lambda c: sum(c.text.lower().count(t) for t in terms),
            reverse=True  # Highest score first
        )

        # Only return chunks that actually match at least one term
        return [c for c in ranked if any(t in c.text.lower() for t in terms)][:k]

    def _load_chunks(self) -> List[PolicyChunk]:
        """
        Reads previously ingested chunks from the JSON storage file.
        Returns an empty list if the file doesn't exist (first run).
        """
        try:
            if not os.path.exists(self._store_path):
                return []  # No policy documents ingested yet — start empty
            with open(self._store_path) as f:
                # Each item in the JSON array is a plain dict — unpack into PolicyChunk
                return [PolicyChunk(**item) for item in json.load(f)]
        except Exception:
            return []  # Corrupted file — start fresh rather than crashing

    def _persist_chunks(self) -> None:
        """
        Serializes all chunks to JSON and writes to disk.

        WHY NOT A DATABASE:
          JSON is human-inspectable, zero-dependency, and fast enough for
          a few thousand chunks. If this grows to tens of thousands of chunks,
          a proper vector database would be the right upgrade.
        """
        # Create the data/ directory if it doesn't exist yet
        os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
        with open(self._store_path, "w") as f:
            # Store as a list of plain dicts so it's human-readable in the file
            json.dump(
                [{"source": c.source, "page": c.page, "text": c.text} for c in self._chunks],
                f,
                indent=2,  # Pretty-print for readability when inspecting manually
            )


# Single shared instance — imported by main.py for /ingest-policy and /generate-plan routes
policy_decoder = PolicyDecoder()
