"""
Build an action-item dataset for LLM training/evaluation.

Reads:
  - iclr_merged_diffs.jsonl: annotated comment/response pairs grouped by paper
  - ICLR dialogues JSON: reviewer-author dialogues per paper

Outputs a JSONL file where each line is one paper with:
  - the full reviewer-author dialogues (input context)
  - the ground-truth action items (target for LLM generation)
"""
import json
import argparse
import os
from collections import defaultdict


def load_merged_diffs(path: str) -> dict[str, list[dict]]:
    """Load merged diffs and group action items by paper_id."""
    paper_actions = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            paper_id = entry['paper_id']
            paper_actions[paper_id].append({
                'comment': entry.get('filtered_comment', entry.get('comment', '')),
                'response': entry.get('filtered_response', ''),
                'diffs': entry.get('diffs', []),
            })
    return paper_actions


def load_dialogues(path: str) -> dict:
    """Load the dialogues JSON keyed by paper ID."""
    with open(path) as f:
        return json.load(f)


def build_dataset(merged_path: str, dialogues_path: str, output_path: str):
    paper_actions = load_merged_diffs(merged_path)
    dialogues = load_dialogues(dialogues_path)

    count = 0
    skipped = 0

    with open(output_path, 'w') as out:
        for paper_id, action_items in paper_actions.items():
            paper_dialogues = dialogues.get(paper_id, {}).get('dialogue', [])
            if not paper_dialogues:
                skipped += 1
                continue

            record = {
                'paper_id': paper_id,
                'dialogues': paper_dialogues,
                'action_items': action_items,
            }
            out.write(json.dumps(record) + '\n')
            count += 1

    print(f"Wrote {count} papers to {output_path}")
    print(f"Skipped {skipped} papers (no dialogues found)")
    print(f"Total action items: {sum(len(v) for v in paper_actions.values())}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build action-item dataset from merged diffs and dialogues.")
    parser.add_argument('--merged-diffs', type=str, default='iclr_merged_diffs.jsonl',
                        help='Path to the merged diffs JSONL file')
    parser.add_argument('--dialogues', type=str,
                        default='ICLR.cc/2024/Conference/dialogues/papers_20260313_224048_with_arxiv_with_pdfs_dialogues.json',
                        help='Path to the dialogues JSON file')
    parser.add_argument('--output', type=str, default='action_item_dataset.jsonl',
                        help='Output JSONL file path')
    args = parser.parse_args()

    build_dataset(args.merged_diffs, args.dialogues, args.output)
