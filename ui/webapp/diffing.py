"""Word-level PDF diff computation for the browser annotation UI.

Ported from the Tk viewer (``legacy/PDF-Diff-Functions/pdf_viewer_clickable.py``)
so the web app runs without any GUI toolkit.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path

import pymupdf

LINE_TOLERANCE_Y = 3
CONTEXT_WORDS = 5
RECT_MERGE_TOLERANCE = 10

_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)

DIFF_TYPE_BY_COLOR = {"red": "deletion", "green": "insertion", "blue": "moved"}


def extract_words_and_pages(pdf_path: str, ignore_ligatures: bool = True) -> tuple[list[dict], list[dict]]:
    """Extract every word of a PDF in reading order, plus per-page dimensions."""
    document = pymupdf.open(pdf_path)
    all_words: list[dict] = []
    pages: list[dict] = []

    try:
        for page_num, page in enumerate(document):
            page.remove_rotation()
            pages.append({"width": round(page.rect.width, 2), "height": round(page.rect.height, 2)})

            raw_words = page.get_text("words", flags=0) if ignore_ligatures else page.get_text("words")

            top_left_in_block: dict[int, tuple[float, float]] = {}
            grouped_lines: list[dict] = []

            for word_info in raw_words:
                x0, y0, x1, y1, _text, block_no = word_info[:6]
                word_center_y = (y0 + y1) / 2

                if block_no not in top_left_in_block:
                    top_left_in_block[block_no] = (x0, y0)
                else:
                    block_x, block_y = top_left_in_block[block_no]
                    if y0 < block_y or (y0 == block_y and x0 < block_x):
                        top_left_in_block[block_no] = (x0, y0)

                added_to_existing_line = False
                for line_group in grouped_lines:
                    if (
                        abs(line_group["y_center"] - word_center_y) < LINE_TOLERANCE_Y
                        and line_group["block_no"] == block_no
                    ):
                        line_group["words"].append(word_info)
                        line_group["y_center"] = sum(
                            (w[1] + w[3]) / 2 for w in line_group["words"]
                        ) / len(line_group["words"])
                        added_to_existing_line = True
                        break

                if not added_to_existing_line:
                    grouped_lines.append(
                        {"y_center": word_center_y, "words": [word_info], "block_no": block_no}
                    )

            grouped_lines.sort(
                key=lambda lg: (
                    top_left_in_block[lg["block_no"]][1],
                    top_left_in_block[lg["block_no"]][0],
                    lg["y_center"],
                )
            )

            for line_group in grouped_lines:
                line_group["words"].sort(key=lambda w: w[0])
                for x0, y0, x1, y1, text, *_rest in line_group["words"]:
                    all_words.append(
                        {
                            "text": text,
                            "x0": round(x0, 2),
                            "y0": round(y0, 2),
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "page_num": page_num,
                            "highlight_color": None,
                        }
                    )
    finally:
        document.close()

    return all_words, pages


def _comparison_keys(words: list[dict], case_insensitive: bool, ignore_quotes: bool) -> list[str]:
    keys = [word["text"] for word in words]
    if case_insensitive:
        keys = [key.lower() for key in keys]
    if ignore_quotes:
        keys = [key.translate(_QUOTE_TRANSLATION) for key in keys]
    return keys


def align_words(
    words_left: list[dict],
    words_right: list[dict],
    case_insensitive: bool = True,
    ignore_quotes: bool = True,
) -> None:
    """Mark deleted words red and inserted words green, in place."""
    left_keys = _comparison_keys(words_left, case_insensitive, ignore_quotes)
    right_keys = _comparison_keys(words_right, case_insensitive, ignore_quotes)

    for word in words_left:
        word["highlight_color"] = None
    for word in words_right:
        word["highlight_color"] = None

    matcher = difflib.SequenceMatcher(None, left_keys, right_keys)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            for index in range(i1, i2):
                words_left[index]["highlight_color"] = "red"
        if tag in ("insert", "replace"):
            for index in range(j1, j2):
                words_right[index]["highlight_color"] = "green"


def _merge_rects(words: list[dict]) -> list[dict]:
    """Merge word boxes that sit on the same line into a single rectangle."""
    merged: list[dict] = []
    current: dict | None = None

    for word in words:
        box = {
            "page": word["page_num"],
            "x0": word["x0"],
            "y0": word["y0"],
            "x1": word["x1"],
            "y1": word["y1"],
        }
        if current is None:
            current = box
            continue

        same_line = (
            current["page"] == box["page"]
            and abs(current["y0"] - box["y0"]) < RECT_MERGE_TOLERANCE
            and abs(current["y1"] - box["y1"]) < RECT_MERGE_TOLERANCE
            and box["x0"] <= current["x1"] + RECT_MERGE_TOLERANCE
        )
        if same_line:
            current["x0"] = min(current["x0"], box["x0"])
            current["y0"] = min(current["y0"], box["y0"])
            current["x1"] = max(current["x1"], box["x1"])
            current["y1"] = max(current["y1"], box["y1"])
        else:
            merged.append(current)
            current = box

    if current is not None:
        merged.append(current)
    return merged


def build_groups(words: list[dict], side: str) -> list[dict]:
    """Collapse runs of consecutive highlighted words into selectable diff groups."""
    prefix = "L" if side == "left" else "R"
    groups: list[dict] = []
    index = 0

    while index < len(words):
        color = words[index]["highlight_color"]
        if not color:
            index += 1
            continue

        end = index
        while end + 1 < len(words) and words[end + 1]["highlight_color"] == color:
            end += 1

        span = words[index : end + 1]
        context_before = " ".join(w["text"] for w in words[max(0, index - CONTEXT_WORDS) : index])
        context_after = " ".join(
            w["text"] for w in words[end + 1 : min(len(words), end + 1 + CONTEXT_WORDS)]
        )
        rects = _merge_rects(span)

        groups.append(
            {
                "id": f"{prefix}{len(groups)}",
                "side": side,
                "diff_type": DIFF_TYPE_BY_COLOR.get(color, "moved"),
                "page_num": span[0]["page_num"],
                "text": " ".join(w["text"] for w in span),
                "context_before": context_before,
                "context_after": context_after,
                "word_count": len(span),
                "rects": rects,
            }
        )
        index = end + 1

    groups.sort(key=lambda g: (g["rects"][0]["page"], g["rects"][0]["y0"], g["rects"][0]["x0"]))
    for position, group in enumerate(groups):
        group["order"] = position
    return groups


def compute_diff(left_pdf: str, right_pdf: str, ignore_ligatures: bool = True) -> dict:
    """Compare two PDFs and return the page geometry plus selectable diff groups."""
    words_left, pages_left = extract_words_and_pages(left_pdf, ignore_ligatures)
    words_right, pages_right = extract_words_and_pages(right_pdf, ignore_ligatures)

    align_words(words_left, words_right)

    return {
        "left": {
            "file_name": os.path.basename(left_pdf),
            "pages": pages_left,
            "groups": build_groups(words_left, "left"),
        },
        "right": {
            "file_name": os.path.basename(right_pdf),
            "pages": pages_right,
            "groups": build_groups(words_right, "right"),
        },
    }


def cache_key(left_pdf: str, right_pdf: str) -> str:
    """Fingerprint the PDF pair so cached diffs are invalidated when a file changes."""
    parts = []
    for path in (left_pdf, right_pdf):
        stat = os.stat(path)
        parts.append(f"{os.path.abspath(path)}|{stat.st_size}|{int(stat.st_mtime)}")
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]


def compute_diff_cached(left_pdf: str, right_pdf: str, cache_dir: Path) -> dict:
    """Return the diff for a PDF pair, reusing an on-disk result when available."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key(left_pdf, right_pdf)}.json"

    if cache_file.exists():
        try:
            with cache_file.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            cache_file.unlink(missing_ok=True)

    diff = compute_diff(left_pdf, right_pdf)
    try:
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(diff, handle, ensure_ascii=False)
    except OSError:
        pass
    return diff
