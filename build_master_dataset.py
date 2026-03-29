"""
Build the consolidated master dataset for diff classification.

One JSONL line per paper, containing:
  - paper_id
  - dialogue: formatted reviewer-author dialogue string
  - action_items: list of {filtered_comment, filtered_response, relevant_diff_indices}
  - all_diffs: full list of paper diffs (each with a diff_index field)

Then splits into train/val/test by paper (using the same paper-to-split
assignments as the action-item splits if they exist, otherwise fresh splits).

Pipeline:
  1. Load iclr_merged_diffs.jsonl → group action items by paper
  2. Load dialogues JSON → get reviewer-author dialogue per paper
  3. Load full paper diffs JSON → get all PDF diffs per paper
  4. Match annotated diffs to paper diffs → get relevant_diff_indices
  5. Join everything into a per-paper master record
  6. Split into train/val/test by paper
  7. Write to Master_DS_ICLR/{master.jsonl, train.jsonl, val.jsonl, test.jsonl}

Usage:
    python build_master_dataset.py

    python build_master_dataset.py \
        --merged-diffs iclr_merged_diffs.jsonl \
        --dialogues ICLR.cc/2024/Conference/dialogues/papers_20260313_224048_with_arxiv_with_pdfs_dialogues.json \
        --paper-diffs ICLR.cc/2024/Conference/diffs/papers_20260313_224048_with_arxiv_with_pdfs_diffs.json \
        --output-dir Master_DS_ICLR

    # For NeurIPS (when data is ready):
    python build_master_dataset.py \
        --merged-diffs neurips_merged_diffs.jsonl \
        --dialogues NeurIPS.cc/2024/Conference/dialogues/papers_20260312_222512_with_arxiv_with_pdfs_dialogues.json \
        --paper-diffs NeurIPS.cc/2024/Conference/diffs/papers_20260312_222512_with_arxiv_with_pdfs_diffs.json \
        --existing-splits neurips_splits \
        --output-dir Master_DS_NeurIPS
"""
import json
import argparse
import glob
import os
import random
from collections import defaultdict
from build_splits import format_dialogue


# ── Data loading ─────────────────────────────────────────────────────────────


def load_merged_diffs(path: str) -> dict[str, list[dict]]:
    """Load iclr_merged_diffs.jsonl and group entries by paper_id.

    Returns {paper_id: [entry, ...]} where each entry has
    filtered_comment, filtered_response, diffs (annotated), etc.
    """
    papers = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            papers[entry['paper_id']].append(entry)
    return dict(papers)


def load_dialogues(path: str) -> dict:
    """Load the dialogues JSON. Returns {paper_id: paper_data}."""
    with open(path) as f:
        return json.load(f)


def load_paper_diffs(path: str) -> dict[str, list[dict]]:
    """Load the full paper diffs JSON. Returns {paper_id: [diff, ...]}."""
    with open(path) as f:
        data = json.load(f)
    return {pid: pdata['diffs'] for pid, pdata in data.items()}


def load_split_assignments(splits_dir: str) -> dict[str, str]:
    """Load paper_id -> split name mapping from existing splits."""
    paper_to_split = {}
    for split_name in ['train', 'val', 'test']:
        path = os.path.join(splits_dir, f'{split_name}.jsonl')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    paper_to_split[entry['paper_id']] = split_name
    return paper_to_split


# ── Diff matching ────────────────────────────────────────────────────────────


def match_annotated_diff_to_paper_diff(
    ann_diff: dict,
    paper_diffs: list[dict],
) -> int | None:
    """Match an annotated diff to its index in the full paper diffs list.

    Annotated diffs have: pane, diff_type, diff_text, context_before,
                          context_after, page_num
    Paper diffs have:     tag, text_pdf1, text_pdf2, context_before/after_pdf1/pdf2,
                          page_nums_pdf1/pdf2, is_moved

    Returns the index into paper_diffs, or None if no match found.
    """
    ann_text = ann_diff.get('diff_text', '').strip()
    ann_ctx_before = ann_diff.get('context_before', '').strip()
    ann_ctx_after = ann_diff.get('context_after', '').strip()
    ann_page = ann_diff.get('page_num')
    ann_pane = ann_diff.get('pane', 'right')
    ann_type = ann_diff.get('diff_type', '')

    if ann_pane == 'right':
        text_key = 'text_pdf2'
        ctx_before_key = 'context_before_pdf2'
        ctx_after_key = 'context_after_pdf2'
        page_key = 'page_nums_pdf2'
    else:
        text_key = 'text_pdf1'
        ctx_before_key = 'context_before_pdf1'
        ctx_after_key = 'context_after_pdf1'
        page_key = 'page_nums_pdf1'

    best_idx = None
    best_score = -1

    for i, pdiff in enumerate(paper_diffs):
        score = 0
        p_text = pdiff.get(text_key, '').strip()
        p_ctx_before = pdiff.get(ctx_before_key, '').strip()
        p_ctx_after = pdiff.get(ctx_after_key, '').strip()
        p_pages = pdiff.get(page_key, [])

        if ann_text and p_text and ann_text in p_text:
            score += 10
        elif ann_text and p_text and p_text in ann_text:
            score += 8

        if ann_ctx_before and p_ctx_before:
            if ann_ctx_before in p_ctx_before or p_ctx_before in ann_ctx_before:
                score += 3
        if ann_ctx_after and p_ctx_after:
            if ann_ctx_after in p_ctx_after or p_ctx_after in ann_ctx_after:
                score += 3

        if ann_page is not None and p_pages:
            if (ann_page - 1) in p_pages or ann_page in p_pages:
                score += 2

        tag = pdiff.get('tag', '')
        is_moved = pdiff.get('is_moved', False)
        if ann_type == 'insertion' and tag in ('insert', 'replace') and not is_moved:
            score += 1
        elif ann_type == 'deletion' and tag in ('delete', 'replace') and not is_moved:
            score += 1
        elif ann_type == 'moved' and is_moved:
            score += 1

        if score > best_score:
            best_score = score
            best_idx = i

    if best_score >= 10:
        return best_idx
    return None


def strip_word_positions(diff: dict) -> dict:
    """Remove verbose word-level bounding-box data to reduce output size."""
    d = dict(diff)
    d.pop('words_pdf1', None)
    d.pop('words_pdf2', None)
    return d


def find_pdf_paths(paper_id: str, pdf_dir: str) -> dict[str, str | None]:
    """Find original and revised PDF paths for a paper.

    Returns {'pdf_original': path_or_None, 'pdf_revised': path_or_None}.
    Convention: *_arxiv_*.pdf = original (pdf1), *_openreview.pdf = revised (pdf2).
    """
    matches = glob.glob(os.path.join(pdf_dir, f'{paper_id}_*'))
    result = {'pdf_original': None, 'pdf_revised': None}
    for p in matches:
        if '_openreview.pdf' in p:
            result['pdf_revised'] = p
        elif '_arxiv_' in p:
            result['pdf_original'] = p
    return result


# ── Master dataset builder ───────────────────────────────────────────────────


def build_master_dataset(
    merged_diffs_path: str,
    dialogues_path: str,
    paper_diffs_path: str,
    output_dir: str,
    pdf_dir: str = '',
    existing_splits_dir: str | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    strip_positions: bool = True,
):
    """Build the full master dataset and split it."""

    # ── Load all data sources ──
    print("Loading data sources...")
    merged_entries = load_merged_diffs(merged_diffs_path)
    dialogues = load_dialogues(dialogues_path)
    all_paper_diffs = load_paper_diffs(paper_diffs_path)

    print(f"  Merged diffs: {sum(len(v) for v in merged_entries.values())} entries "
          f"across {len(merged_entries)} papers")
    print(f"  Dialogues: {len(dialogues)} papers")
    print(f"  Paper diffs: {len(all_paper_diffs)} papers "
          f"({sum(1 for v in all_paper_diffs.values() if v)} with diffs)")

    # ── Load existing splits if available ──
    paper_to_split = {}
    if existing_splits_dir and os.path.isdir(existing_splits_dir):
        paper_to_split = load_split_assignments(existing_splits_dir)
        print(f"  Existing split assignments: {len(paper_to_split)} papers")

    # ── Build per-paper records ──
    print("\nBuilding master dataset...")
    master = []
    stats = {
        'papers_processed': 0,
        'papers_skipped_no_dialogue': 0,
        'papers_skipped_no_diffs': 0,
        'papers_output': 0,
        'total_action_items': 0,
        'action_items_with_annotated_diffs': 0,
        'action_items_matched': 0,
        'total_annotated_diffs': 0,
        'total_matched_diffs': 0,
    }

    for paper_id, entries in merged_entries.items():
        stats['papers_processed'] += 1

        # Get dialogue
        paper_dialogue_data = dialogues.get(paper_id, {}).get('dialogue', [])
        if not paper_dialogue_data:
            stats['papers_skipped_no_dialogue'] += 1
            continue

        # Format dialogue (keep both raw JSON and formatted string)
        dialogue_str = format_dialogue(paper_dialogue_data)

        # Get full diffs for this paper
        full_diffs = all_paper_diffs.get(paper_id, [])
        if not full_diffs:
            stats['papers_skipped_no_diffs'] += 1
            continue

        # Add diff_index to each diff, optionally strip word positions
        indexed_diffs = []
        for idx, d in enumerate(full_diffs):
            dd = strip_word_positions(d) if strip_positions else dict(d)
            dd['diff_index'] = idx
            indexed_diffs.append(dd)

        # Process each action item for this paper
        action_items = []
        for entry in entries:
            stats['total_action_items'] += 1
            annotated_diffs = entry.get('diffs', [])

            # Match annotated diffs to paper diffs
            relevant_indices = []
            if annotated_diffs:
                stats['action_items_with_annotated_diffs'] += 1
                for ann_diff in annotated_diffs:
                    stats['total_annotated_diffs'] += 1
                    idx = match_annotated_diff_to_paper_diff(ann_diff, full_diffs)
                    if idx is not None:
                        relevant_indices.append(idx)
                        stats['total_matched_diffs'] += 1

            relevant_indices = sorted(set(relevant_indices))
            if relevant_indices:
                stats['action_items_matched'] += 1

            action_items.append({
                'filtered_comment': entry.get('filtered_comment', entry.get('comment', '')),
                'filtered_response': entry.get('filtered_response', ''),
                'relevant_diff_indices': relevant_indices,
            })

        # Find PDF paths
        pdf_paths = find_pdf_paths(paper_id, pdf_dir) if pdf_dir else {}

        stats['papers_output'] += 1
        record = {
            'paper_id': paper_id,
            'dialogue': dialogue_str,
            'dialogue_json': paper_dialogue_data,
            'action_items': action_items,
            'all_diffs': indexed_diffs,
        }
        if pdf_paths.get('pdf_original') or pdf_paths.get('pdf_revised'):
            record['pdf_original'] = pdf_paths.get('pdf_original')
            record['pdf_revised'] = pdf_paths.get('pdf_revised')
        master.append(record)

    # ── Print stats ──
    print(f"\n{'=' * 60}")
    print(f"Master Dataset Stats")
    print(f"{'=' * 60}")
    print(f"Papers processed:                {stats['papers_processed']}")
    print(f"  skipped (no dialogue):         {stats['papers_skipped_no_dialogue']}")
    print(f"  skipped (no paper diffs):      {stats['papers_skipped_no_diffs']}")
    print(f"  output:                        {stats['papers_output']}")
    print(f"Total action items:              {stats['total_action_items']}")
    print(f"  with annotated diffs:          {stats['action_items_with_annotated_diffs']}")
    print(f"  matched to paper diffs:        {stats['action_items_matched']}")
    print(f"Total annotated diffs:           {stats['total_annotated_diffs']}")
    print(f"  successfully matched:          {stats['total_matched_diffs']}")

    # ── Write master JSONL ──
    os.makedirs(output_dir, exist_ok=True)
    master_path = os.path.join(output_dir, 'master.jsonl')
    with open(master_path, 'w') as f:
        for record in master:
            f.write(json.dumps(record) + '\n')
    print(f"\nWrote master dataset: {len(master)} papers -> {master_path}")

    # ── Split by paper ──
    if paper_to_split:
        # Use existing assignments
        split_data = defaultdict(list)
        unassigned = []
        for record in master:
            s = paper_to_split.get(record['paper_id'])
            if s:
                split_data[s].append(record)
            else:
                unassigned.append(record)

        if unassigned:
            print(f"\nWARNING: {len(unassigned)} papers not in existing splits — "
                  f"assigning to train")
            split_data['train'].extend(unassigned)
    else:
        # Fresh splits
        random.seed(seed)
        papers = list(master)
        random.shuffle(papers)
        n = len(papers)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        split_data = {
            'train': papers[:n_train],
            'val': papers[n_train:n_train + n_val],
            'test': papers[n_train + n_val:],
        }

    # Shuffle within each split
    random.seed(seed)
    for s in split_data:
        random.shuffle(split_data[s])

    # Write splits
    print(f"\nSplits:")
    for split_name in ['train', 'val', 'test']:
        records = split_data.get(split_name, [])
        path = os.path.join(output_dir, f'{split_name}.jsonl')
        with open(path, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
        n_papers = len(records)
        n_items = sum(len(r['action_items']) for r in records)
        n_with_diffs = sum(
            1 for r in records for ai in r['action_items']
            if ai['relevant_diff_indices']
        )
        print(f"  {split_name}: {n_papers} papers, {n_items} action items "
              f"({n_with_diffs} with matched diffs) -> {path}")

    print(f"\nDone. Output in {output_dir}/")


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build consolidated master dataset with diffs, dialogues, and action items."
    )
    parser.add_argument(
        '--merged-diffs', type=str, default='iclr_merged_diffs.jsonl',
        help='Path to merged annotation diffs JSONL',
    )
    parser.add_argument(
        '--dialogues', type=str,
        default='ICLR.cc/2024/Conference/dialogues/papers_20260313_224048_with_arxiv_with_pdfs_dialogues.json',
        help='Path to dialogues JSON',
    )
    parser.add_argument(
        '--paper-diffs', type=str,
        default='ICLR.cc/2024/Conference/diffs/papers_20260313_224048_with_arxiv_with_pdfs_diffs.json',
        help='Path to full paper diffs JSON',
    )
    parser.add_argument(
        '--pdf-dir', type=str,
        default='ICLR.cc/2024/Conference/PDFs',
        help='Directory containing PDF files (original and revised)',
    )
    parser.add_argument(
        '--existing-splits', type=str, default='',
        help='Directory with existing action-item splits (for paper-to-split assignments). '
             'Leave empty (default) to create fresh splits with --seed.',
    )
    parser.add_argument(
        '--output-dir', type=str, default='Master_DS_ICLR',
        help='Output directory',
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.7,
        help='Train ratio (only used if no existing splits)',
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.15,
        help='Val ratio (only used if no existing splits)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--keep-word-positions', action='store_true',
        help='Keep word-level bounding boxes in diffs (makes files much larger)',
    )
    args = parser.parse_args()

    existing_splits = args.existing_splits if args.existing_splits else None

    build_master_dataset(
        merged_diffs_path=args.merged_diffs,
        dialogues_path=args.dialogues,
        paper_diffs_path=args.paper_diffs,
        output_dir=args.output_dir,
        pdf_dir=args.pdf_dir,
        existing_splits_dir=existing_splits,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        strip_positions=not args.keep_word_positions,
    )
