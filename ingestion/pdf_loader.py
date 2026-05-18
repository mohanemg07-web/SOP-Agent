"""
SOP PDF Loader — parses PDF files into section-aware chunks.

Uses pdfplumber for extraction and regex-based heuristics for
section boundary detection. Falls back to paragraph splitting
when no sections are detected.
"""

import logging
import re

import pdfplumber

logger = logging.getLogger(__name__)


class SOPLoader:
    """Load and chunk an SOP PDF into structured sections."""

    # Heuristic patterns ordered by priority
    _PATTERNS = [
        re.compile(r"^(Step|STEP)\s+\d+", re.IGNORECASE),  # "Step 3: ..."
        re.compile(r"^\d+[\.\)]\s+[A-Z]{2,}"),              # "3. APPROVE" (ALL CAPS only, not "3. Gender")
        re.compile(r"^[A-Z\s]{10,}$"),                       # ALL CAPS lines > 10 chars
    ]

    # Headings that are structural/navigational and should never be treated as steps
    _EXCLUDED_HEADINGS = {
        "table of contents",
        "contents",
        "index",
        "references",
        "appendix",
        "revision history",
        "document history",
        "change log",
    }

    def __init__(self, pdf_path: str):
        """Initialise with the path to an SOP PDF file.

        Args:
            pdf_path: Absolute or relative path to the PDF.
        """
        self.pdf_path = pdf_path
        self._chunks_cache: list[dict] | None = None  # cache to avoid re-parsing

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def load_and_chunk(self) -> list[dict]:
        """Parse the PDF and return a list of section chunk dicts.

        Each dict has:
            chunk_id      – e.g. "step_3"
            section_title – the heading line
            content       – full text of the section
            page          – page number where the section starts
            step_number   – integer step index

        Results are cached so subsequent calls do not re-parse the PDF.

        Returns:
            List of chunk dicts. Empty list on error.
        """
        # Return cached result to avoid re-parsing and duplicate steps
        if self._chunks_cache is not None:
            return self._chunks_cache

        try:
            pages = self._extract_pages()
        except FileNotFoundError:
            logger.error("PDF not found: %s", self.pdf_path)
            return []
        except Exception as exc:
            logger.error("Failed to read PDF %s: %s", self.pdf_path, exc)
            return []

        if not pages:
            logger.warning("PDF is empty or could not be read: %s", self.pdf_path)
            return []

        # Build a flat list of (line, page_number) tuples
        lines_with_pages: list[tuple[str, int]] = []
        for page_num, page_text in pages:
            for line in page_text.split("\n"):
                lines_with_pages.append((line, page_num))

        # Detect section boundaries
        boundaries = self._detect_boundaries(lines_with_pages)

        if boundaries:
            self._chunks_cache = self._build_chunks_from_boundaries(lines_with_pages, boundaries)
        else:
            # Fallback: split by double newlines into ≤500-char chunks
            logger.info("No section boundaries found — using paragraph fallback.")
            self._chunks_cache = self._fallback_chunking(pages)

        return self._chunks_cache

    def extract_step_list(self) -> list[dict]:
        """Return a simplified list of step metadata.

        Each dict has: id, title, description (first 200 chars of content).
        """
        chunks = self.load_and_chunk()
        step_list = []
        for chunk in chunks:
            step_list.append({
                "id": chunk["chunk_id"],
                "title": chunk["section_title"],
                "description": chunk["content"][:200],
            })
        return step_list

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _extract_pages(self) -> list[tuple[int, str]]:
        """Extract text from every page of the PDF.

        Returns:
            List of (page_number, page_text) tuples. Page numbers are 1-based.
        """
        pages: list[tuple[int, str]] = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append((i, text))
        except Exception as exc:
            logger.error("pdfplumber failed on %s: %s", self.pdf_path, exc)
            raise
        return pages

    # Matches TOC dot-leader lines like "1. BACKGROUND ............" or "Purpose ..........."
    _TOC_LEADER_PATTERN = re.compile(r"\.{3,}\s*\d*\s*$")

    def _detect_boundaries(
        self, lines_with_pages: list[tuple[str, int]]
    ) -> list[tuple[int, str, int]]:
        """Find section boundary indices using heuristic patterns.

        TOC lines (those ending with dot leaders like ".......") are
        explicitly excluded so the Table of Contents is never mistaken
        for real section headings.

        Returns:
            List of (line_index, heading_text, page_number) tuples.
        """
        boundaries: list[tuple[int, str, int]] = []
        seen_titles: set[str] = set()  # deduplicate headings with identical text
        total_lines = len(lines_with_pages)

        for idx, (line, page_num) in enumerate(lines_with_pages):
            stripped = line.strip()
            if not stripped:
                continue

            # Skip TOC dot-leader lines (e.g. "1. BACKGROUND ............")
            if self._TOC_LEADER_PATTERN.search(stripped):
                continue

            # Skip known structural/navigational headings that are never steps
            if stripped.lower() in self._EXCLUDED_HEADINGS:
                continue

            # Priority a-c: regex patterns
            matched = False
            for pattern in self._PATTERNS:
                if pattern.match(stripped):
                    # Deduplicate: skip if we already have this exact heading
                    norm = stripped.lower()
                    if norm not in seen_titles:
                        boundaries.append((idx, stripped, page_num))
                        seen_titles.add(norm)
                    matched = True
                    break

            if not matched:
                # Priority d: short line followed by a blank line
                if (
                    len(stripped) < 80
                    and idx + 1 < total_lines
                    and lines_with_pages[idx + 1][0].strip() == ""
                ):
                    # Only treat as boundary if it looks like a heading
                    # (starts with uppercase letter and is not a sentence fragment)
                    if stripped[0].isupper() and not stripped.endswith(","):
                        norm = stripped.lower()
                        if norm not in seen_titles:
                            boundaries.append((idx, stripped, page_num))
                            seen_titles.add(norm)

        return boundaries

    def _build_chunks_from_boundaries(
        self,
        lines_with_pages: list[tuple[str, int]],
        boundaries: list[tuple[int, str, int]],
    ) -> list[dict]:
        """Group text between consecutive boundaries into chunks."""
        chunks: list[dict] = []

        for i, (start_idx, title, page_num) in enumerate(boundaries):
            # Content runs from start_idx to the next boundary (or end)
            end_idx = (
                boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines_with_pages)
            )
            content_lines = [
                lines_with_pages[j][0] for j in range(start_idx, end_idx)
            ]
            content = "\n".join(content_lines).strip()

            step_number = i + 1
            chunk_id = f"step_{step_number}"

            chunks.append({
                "chunk_id": chunk_id,
                "section_title": title,
                "content": content,
                "page": page_num,
                "step_number": step_number,
            })

        return chunks

    def _fallback_chunking(self, pages: list[tuple[int, str]]) -> list[dict]:
        """Split full document text by double newlines into ≤500-char chunks."""
        full_text = "\n\n".join(text for _, text in pages)
        paragraphs = re.split(r"\n{2,}", full_text)

        chunks: list[dict] = []
        current_chunk = ""
        chunk_counter = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 > 500 and current_chunk:
                chunk_counter += 1
                chunks.append(self._make_fallback_chunk(chunk_counter, current_chunk))
                current_chunk = para
            else:
                current_chunk = f"{current_chunk}\n\n{para}".strip() if current_chunk else para

        # Flush remaining
        if current_chunk:
            chunk_counter += 1
            chunks.append(self._make_fallback_chunk(chunk_counter, current_chunk))

        return chunks

    @staticmethod
    def _make_fallback_chunk(number: int, content: str) -> dict:
        """Create a chunk dict for the fallback splitting strategy."""
        return {
            "chunk_id": f"section_{number}",
            "section_title": f"Section {number}",
            "content": content,
            "page": 1,
            "step_number": number,
        }
