# Diff Linking Annotation UI

A browser-based tool for linking reviewer/author action items to the concrete changes
between the original (arXiv) and revised (OpenReview) version of a paper.

The annotator sees the two PDFs side by side with every textual change highlighted —
deletions in red on the left, insertions in green on the right — plus the action item
being annotated. Clicking a highlight records that change against the action item.

Everything renders in the browser. There is no desktop toolkit, no Tk, no X server and
no display requirement, so it runs the same on Windows (including ARM64 / NPU laptops),
macOS, Linux and WSL. You can also run the server on one machine and annotate from
another over the network.

This replaces the earlier Tk desktop viewer, which needed `tkinterdnd2`, `klembord` and
a graphical display.

## Setup

Requires Python 3.10+. From this directory:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

**macOS / Linux / WSL**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Only two dependencies are installed: `Flask` and `PyMuPDF`.

## Run

```bash
python run.py
```

With no arguments this opens the bundled ten-item example batch, and a browser tab opens
at <http://127.0.0.1:8000/> automatically.

For a real batch:

```bash
python run.py path/to/batch.jsonl path/to/pdfs --output path/to/linked_diffs.jsonl
```

| Option | Description |
| --- | --- |
| `--output`, `-o` | Output JSONL (default `<input>_with_diffs.jsonl`) |
| `--port` | Port to listen on (default `8000`) |
| `--host` | Bind address; use `0.0.0.0` to let others on your network annotate |
| `--no-browser` | Do not open a browser automatically |

Input entries need `paper_id`, `filtered_comment` and `filtered_response`. PDFs are
matched by name: `{paper_id}_arxiv_*.pdf` (original) and `{paper_id}_openreview.pdf`
(revised).

If the process exits immediately, something else is already using port 8000 — start it
on another port with `python run.py --port 8010`.

## Annotate

1. Read the action item in the right-hand panel.
2. **Click** a highlighted change to select it; click again to deselect. **Drag a box**
   over an area to select every change inside it.
3. **Next** / **Prev** jump between changes; **Fit** and **+ / −** control zoom; **Sync
   scroll** keeps both PDFs aligned.
4. **Save & Next** writes the entry with its selected changes and loads the next action
   item. **Skip** records the entry with no changes.

Keyboard: `N` / `P` next and previous change, `F` fit width, `+` / `-` zoom,
`S` save & next, `Esc` clear the selection.

## Output

Each input entry is appended to the output JSONL with a `diffs` array, in the same shape
the previous desktop tool produced:

```json
{
  "paper_id": "xEJMoj1SpX",
  "unique_id": "entry_00096_cc88a591",
  "filtered_comment": "...",
  "filtered_response": "...",
  "diffs": [
    {
      "timestamp": "2026-08-21T15:59:26.591059",
      "pane": "right",
      "file_name": "xEJMoj1SpX_openreview.pdf",
      "diff_type": "insertion",
      "page_num": 1,
      "diff_text": "Utrecht University",
      "context_before": "University a.a.salah@uu.nl Itir Onal Ertugrul",
      "context_after": "i.onalertugrul@uu.nl ABSTRACT Diffusion models have",
      "word_count": 2
    }
  ]
}
```

Entries whose PDFs are missing are recorded automatically with
`"skipped": true, "reason": "pdfs_not_found"` and are never shown to the annotator.
Entries you dismiss with **Skip** are recorded with `"skipped": true` and no `reason`.

Progress is resumable: on start-up the tool counts the rows already in the output file
and continues from the next entry, so you can stop with `Ctrl+C` and pick up later.

Computed diffs are cached under `.cache/diffs/`, keyed by PDF path and modification
time, so revisiting a paper is instant. Deleting that folder forces a recompute.

## Layout

| Path | Purpose |
| --- | --- |
| `run.py` | Command-line entry point that starts the local server |
| `webapp/diffing.py` | Word extraction, alignment and grouping of changes |
| `webapp/session.py` | Entry queue, PDF lookup, page rendering, result writing |
| `webapp/server.py` | Flask routes |
| `webapp/static/`, `webapp/templates/` | Browser UI (vanilla JS, no build step) |
| `examples/` | Ten sample action items with the three papers they refer to |
