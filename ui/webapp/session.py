"""Annotation session state: entry queue, PDF lookup, diff caching and result writing."""

from __future__ import annotations

import datetime
import json
import threading
from pathlib import Path

import pymupdf

from . import diffing

RENDER_SCALES = (1.0, 1.5, 2.0, 3.0, 4.0)
MAX_CACHED_PAGE_IMAGES = 80


class PaperContext:
    """Holds the computed diff and rendered page images for one paper."""

    def __init__(self, paper_id: str, left_pdf: Path, right_pdf: Path, cache_dir: Path):
        self.paper_id = paper_id
        self.paths = {"left": left_pdf, "right": right_pdf}
        self.cache_dir = cache_dir
        self.status = "computing"
        self.diff: dict | None = None
        self.error: str | None = None
        self._documents: dict[str, pymupdf.Document] = {}
        self._images: dict[tuple[str, int, float], bytes] = {}
        self._render_lock = threading.Lock()
        self._words: dict[str, list[dict]] | None = None
        self._words_lock = threading.Lock()
        self._thread = threading.Thread(target=self._compute, daemon=True)
        self._thread.start()

    def _compute(self) -> None:
        try:
            self.diff = diffing.compute_diff_cached(
                str(self.paths["left"]), str(self.paths["right"]), self.cache_dir
            )
            self.status = "ready"
        except Exception as exc:  # surfaced to the browser as an error banner
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"

    def render_page(self, side: str, page_number: int, scale: float) -> bytes:
        if side not in self.paths:
            raise KeyError(side)
        scale = min(RENDER_SCALES, key=lambda option: abs(option - scale))
        key = (side, page_number, scale)

        with self._render_lock:
            if key in self._images:
                return self._images[key]

            document = self._documents.get(side)
            if document is None:
                document = pymupdf.open(str(self.paths[side]))
                self._documents[side] = document

            page = document.load_page(page_number)
            page.remove_rotation()
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
            data = pixmap.tobytes("png")

            while len(self._images) >= MAX_CACHED_PAGE_IMAGES:
                self._images.pop(next(iter(self._images)))
            self._images[key] = data
            return data

    def search(self, query: str) -> dict:
        with self._words_lock:
            if self._words is None:
                self._words = {
                    side: diffing.extract_words_and_pages(str(path))[0]
                    for side, path in self.paths.items()
                }
        return {side: diffing.find_matches(words, query) for side, words in self._words.items()}

    def close(self) -> None:
        with self._render_lock:
            for document in self._documents.values():
                document.close()
            self._documents.clear()
            self._images.clear()


class AnnotationSession:
    """Drives the entry queue and appends annotated entries to the output JSONL."""

    def __init__(self, input_jsonl: Path, pdf_dir: Path, output_path: Path, cache_dir: Path):
        self.input_jsonl = input_jsonl
        self.pdf_dir = pdf_dir
        self.output_path = output_path
        self.cache_dir = cache_dir
        self.entries = self._load_entries(input_jsonl)
        self.index = self._count_output_entries(output_path)
        self.auto_skipped: list[dict] = []
        self.paper: PaperContext | None = None
        self._lock = threading.RLock()
        self._prepare_current()

    @staticmethod
    def _load_entries(path: Path) -> list[dict]:
        entries = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    @staticmethod
    def _count_output_entries(path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    continue
        return count

    def find_pdfs(self, paper_id: str) -> tuple[Path | None, Path | None]:
        """Original (arXiv) and revised (OpenReview) PDFs for a paper."""
        arxiv_matches = sorted(self.pdf_dir.glob(f"{paper_id}_arxiv_*.pdf"))
        openreview = self.pdf_dir / f"{paper_id}_openreview.pdf"
        return (
            arxiv_matches[0] if arxiv_matches else None,
            openreview if openreview.exists() else None,
        )

    def _append_result(self, record: dict) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _prepare_current(self) -> None:
        """Load the current entry's paper, auto-recording entries whose PDFs are missing."""
        while self.index < len(self.entries):
            entry = self.entries[self.index]
            paper_id = entry.get("paper_id", "")
            left_pdf, right_pdf = self.find_pdfs(paper_id)

            if not left_pdf or not right_pdf:
                self._append_result(
                    {**entry, "diffs": [], "skipped": True, "reason": "pdfs_not_found"}
                )
                self.auto_skipped.append({"index": self.index, "paper_id": paper_id})
                self.index += 1
                continue

            if self.paper is None or self.paper.paths["left"] != left_pdf or self.paper.paths["right"] != right_pdf:
                if self.paper is not None:
                    self.paper.close()
                self.paper = PaperContext(paper_id, left_pdf, right_pdf, self.cache_dir)
            return

        if self.paper is not None:
            self.paper.close()
            self.paper = None

    def state(self) -> dict:
        with self._lock:
            done = self.index >= len(self.entries)
            entry = None if done else self.entries[self.index]
            paper = self.paper
            return {
                "index": self.index,
                "total": len(self.entries),
                "done": done,
                "entry": entry,
                "paper_id": None if entry is None else entry.get("paper_id"),
                "openreview_url": None
                if entry is None
                else f"https://openreview.net/forum?id={entry.get('paper_id', '')}",
                "diff_status": "idle" if paper is None else paper.status,
                "diff_error": None if paper is None else paper.error,
                "auto_skipped": self.auto_skipped,
                "input_path": str(self.input_jsonl),
                "output_path": str(self.output_path),
            }

    def diff_payload(self) -> tuple[str, dict | None, str | None]:
        with self._lock:
            if self.paper is None:
                return "idle", None, None
            return self.paper.status, self.paper.diff, self.paper.error

    def render_page(self, side: str, page_number: int, scale: float) -> bytes:
        with self._lock:
            paper = self.paper
        if paper is None:
            raise LookupError("no paper loaded")
        return paper.render_page(side, page_number, scale)

    def search(self, query: str) -> dict:
        with self._lock:
            paper = self.paper
        if paper is None:
            raise LookupError("no paper loaded")
        return paper.search(query)

    def _diff_records(self, group_ids: list[str]) -> list[dict]:
        if self.paper is None or not self.paper.diff:
            return []

        groups_by_id = {}
        for side in ("left", "right"):
            for group in self.paper.diff[side]["groups"]:
                groups_by_id[group["id"]] = (side, group)

        timestamp = datetime.datetime.now().isoformat()
        records = []
        for group_id in group_ids:
            match = groups_by_id.get(group_id)
            if match is None:
                continue
            side, group = match
            records.append(
                {
                    "timestamp": timestamp,
                    "pane": side,
                    "file_name": self.paper.diff[side]["file_name"],
                    "diff_type": group["diff_type"],
                    "page_num": group["page_num"] + 1,
                    "diff_text": group["text"],
                    "context_before": group["context_before"],
                    "context_after": group["context_after"],
                    "word_count": group["word_count"],
                }
            )
        return records

    def save_current(self, group_ids: list[str]) -> dict:
        with self._lock:
            if self.index >= len(self.entries):
                return self.state()
            entry = self.entries[self.index]
            self._append_result({**entry, "diffs": self._diff_records(group_ids)})
            self.index += 1
            self._prepare_current()
            return self.state()

    def skip_current(self) -> dict:
        with self._lock:
            if self.index >= len(self.entries):
                return self.state()
            entry = self.entries[self.index]
            self._append_result({**entry, "diffs": [], "skipped": True})
            self.index += 1
            self._prepare_current()
            return self.state()
