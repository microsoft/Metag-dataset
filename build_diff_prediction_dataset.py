"""
Build a diff-prediction dataset for LLM training/evaluation.

Given a reviewer-author dialogue and a single action item, predict which
specific PDF diffs (from the full set of diffs for that paper) correspond
to the change described by the action item.

Reads:
  - action_item_dataset_iclr_master.jsonl: papers with dialogues + action items
    (each action item has annotator-linked diffs)
  - ICLR.cc/2024/Conference/diffs/<diffs_file>.json: full PDF diffs per paper

Outputs split JSONL files (train/val/test) where each line is one
(action_item, paper) pair with:
  - paper_id: the paper identifier
  - dialogue: formatted reviewer-author dialogue string
  - action_item: {comment, response} for one action item
  - all_diffs: list of all diffs for the paper (each with a diff_index field)
  - correct_diff_indices: list of indices into all_diffs that are the ground truth

Paper-to-split assignments are taken from the existing action-item splits
(splits/{train,val,test}.jsonl) to prevent paper leakage across sets.

Usage:
    python build_diff_prediction_dataset.py
    python build_diff_prediction_dataset.py --output-dir diff_prediction_splits
    python build_diff_prediction_dataset.py --strip-word-positions  # smaller output
"""
import json
import argparse
import os
import random
from collections import defaultdict
from build_splits import format_dialogue


def load_master_dataset(path: str) -> list[dict]:
    """Load the master action item dataset (papers with dialogues + action items)."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_paper_diffs(path: str) -> dict[str, list[dict]]:
    """Load the full paper diffs JSON. Returns {paper_id: [diff, ...]}."""
    with open(path) as f:
        data = json.load(f)
    return {pid: pdata['diffs'] for pid, pdata in data.items()}


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
    ann_pane = ann_diff.get('pane', 'right')  # 'right' = pdf2, 'left' = pdf1
    ann_type = ann_diff.get('diff_type', '')

    # Determine which PDF side to match against
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

        # Text match (strongest signal)
        if ann_text and p_text and ann_text in p_text:
            score += 10
        elif ann_text and p_text and p_text in ann_text:
            score += 8

        # Context matches
        if ann_ctx_before and p_ctx_before:
            if ann_ctx_before in p_ctx_before or p_ctx_before in ann_ctx_before:
                score += 3
        if ann_ctx_after and p_ctx_after:
            if ann_ctx_after in p_ctx_after or p_ctx_after in ann_ctx_after:
                score += 3

        # Page number match (0-indexed in paper diffs, 1-indexed in annotated)
        if ann_page is not None and p_pages:
            # Annotated page_num appears to be 1-indexed based on the data
            if (ann_page - 1) in p_pages or ann_page in p_pages:
                score += 2

        # Diff type match
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

    # Require a minimum score to accept the match (at least text match)
    if best_score >= 10:
        return best_idx
    return None


def strip_word_positions(diff: dict) -> dict:
    """Remove verbose word-level bounding-box data to reduce output size."""
    d = dict(diff)
    d.pop('words_pdf1', None)
    d.pop('words_pdf2', None)
    return d


def load_split_assignments(splits_dir: str) -> dict[str, str]:
    """Load paper_id -> split name mapping from existing action-item splits."""
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


def build_dataset(
    master_path: str,
    diffs_path: str,
    output_dir: str,
    splits_dir: str = 'splits',
    strip_positions: bool = False,
    seed: int = 42,
):
    master = load_master_dataset(master_path)
    paper_diffs = load_paper_diffs(diffs_path)
    paper_to_split = load_split_assignments(splits_dir)

    results = []
    stats = {
        'total_action_items': 0,
        'action_items_with_annotated_diffs': 0,
        'action_items_matched': 0,
        'action_items_partial_match': 0,
        'action_items_no_paper_diffs': 0,
        'total_annotated_diffs': 0,
        'total_matched_diffs': 0,
        'papers_processed': 0,
        'papers_with_output': 0,
    }

    for entry in master:
        paper_id = entry['paper_id']
        dialogues = entry.get('dialogues', [])
        action_items = entry.get('action_items', [])

        stats['papers_processed'] += 1

        # Format the dialogue
        dialogue_str = format_dialogue(dialogues)

        # Get full diffs for this paper
        full_diffs = paper_diffs.get(paper_id, [])

        paper_has_output = False

        for ai in action_items:
            stats['total_action_items'] += 1
            annotated_diffs = ai.get('diffs', [])

            if not annotated_diffs:
                continue

            stats['action_items_with_annotated_diffs'] += 1

            if not full_diffs:
                stats['action_items_no_paper_diffs'] += 1
                continue

            # Match each annotated diff to the full paper diffs
            correct_indices = []
            for ann_diff in annotated_diffs:
                stats['total_annotated_diffs'] += 1
                idx = match_annotated_diff_to_paper_diff(ann_diff, full_diffs)
                if idx is not None:
                    correct_indices.append(idx)
                    stats['total_matched_diffs'] += 1

            if not correct_indices:
                stats['action_items_partial_match'] += 1
                continue

            stats['action_items_matched'] += 1
            paper_has_output = True

            # Deduplicate and sort indices
            correct_indices = sorted(set(correct_indices))

            # Prepare the diffs list with diff_index field
            output_diffs = []
            for idx, d in enumerate(full_diffs):
                dd = strip_word_positions(d) if strip_positions else dict(d)
                dd['diff_index'] = idx
                output_diffs.append(dd)

            results.append({
                'paper_id': paper_id,
                'dialogue': dialogue_str,
                'action_item': {
                    'comment': ai['comment'],
                    'response': ai['response'],
                },
                'all_diffs': output_diffs,
                'correct_diff_indices': correct_indices,
            })

        if paper_has_output:
            stats['papers_with_output'] += 1

    # Split results by paper_id using existing split assignments
    split_results = defaultdict(list)
    unassigned = []
    for r in results:
        split_name = paper_to_split.get(r['paper_id'])
        if split_name:
            split_results[split_name].append(r)
        else:
            unassigned.append(r)

    if unassigned:
        print(f"WARNING: {len(unassigned)} entries from {len(set(r['paper_id'] for r in unassigned))} "
              f"papers not in existing splits — adding to train")
        split_results['train'].extend(unassigned)

    # Shuffle within each split for reproducibility
    random.seed(seed)
    for split_name in split_results:
        random.shuffle(split_results[split_name])

    # Write split files
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ['train', 'val', 'test']:
        split_data = split_results.get(split_name, [])
        path = os.path.join(output_dir, f'{split_name}.jsonl')
        with open(path, 'w') as f:
            for r in split_data:
                f.write(json.dumps(r) + '\n')
        n_papers = len(set(r['paper_id'] for r in split_data))
        print(f"{split_name}: {len(split_data)} entries from {n_papers} papers -> {path}")

    # Print stats
    print(f"\n{'=' * 60}")
    print(f"Diff Prediction Dataset Stats")
    print(f"{'=' * 60}")
    print(f"Papers processed:                {stats['papers_processed']}")
    print(f"Papers with output:              {stats['papers_with_output']}")
    print(f"Total action items:              {stats['total_action_items']}")
    print(f"  with annotated diffs:          {stats['action_items_with_annotated_diffs']}")
    print(f"  matched to paper diffs:        {stats['action_items_matched']}")
    print(f"  partial/no match:              {stats['action_items_partial_match']}")
    print(f"  no paper diffs available:      {stats['action_items_no_paper_diffs']}")
    print(f"Total annotated diffs:           {stats['total_annotated_diffs']}")
    print(f"  successfully matched:          {stats['total_matched_diffs']}")
    print(f"Output entries:                  {sum(len(v) for v in split_results.values())}")
    print(f"Wrote to: {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build diff-prediction dataset from action items and paper diffs."
    )
    parser.add_argument(
        '--master', type=str, default='action_item_dataset_iclr_master.jsonl',
        help='Path to the master action item dataset JSONL',
    )
    parser.add_argument(
        '--diffs', type=str,
        default='ICLR.cc/2024/Conference/diffs/papers_20260313_224048_with_arxiv_with_pdfs_diffs.json',
        help='Path to the full paper diffs JSON',
    )
    parser.add_argument(
        '--output-dir', type=str, default='diff_prediction_splits',
        help='Output directory for train/val/test JSONL files',
    )
    parser.add_argument(
        '--splits-dir', type=str, default='splits',
        help='Directory with existing action-item splits (for paper assignments)',
    )
    parser.add_argument(
        '--strip-word-positions', action='store_true',
        help='Remove word-level bounding boxes from diffs to reduce file size',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for shuffling',
    )
    args = parser.parse_args()

    build_dataset(
        master_path=args.master,
        diffs_path=args.diffs,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        strip_positions=args.strip_word_positions,
        seed=args.seed,
    )
