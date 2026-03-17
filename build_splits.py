"""
Build train/val/test splits for action item detection.

Reads action_item_dataset_iclr_master.jsonl and produces three JSONL files
with the reviewer-author dialogue formatted as a readable string (input)
and the action items as structured targets (output).

Schema per line:
{
  "paper_id": "...",
  "input": "<formatted dialogue string with review weaknesses/questions + follow-up>",
  "target": {
    "action_items": [
      {"comment": "...", "response": "..."},
      ...
    ]
  }
}
"""
import json
import argparse
import random


def extract_value(field):
    """Extract text value from OpenReview content fields."""
    if isinstance(field, dict):
        return field.get('value', '')
    return str(field) if field else ''


def format_dialogue(dialogues: list[dict]) -> str:
    """Format all reviewer dialogues into a single readable string."""
    parts = []

    for reviewer_data in dialogues:
        reviewer_id = reviewer_data.get('reviewer_id', 'Unknown')
        review = reviewer_data.get('review', {})
        content = review.get('content', {})

        weaknesses = extract_value(content.get('weaknesses', ''))
        questions = extract_value(content.get('questions', ''))

        # Start the reviewer block
        parts.append(f"## {reviewer_id}")

        if weaknesses:
            parts.append(f"### Weaknesses\n{weaknesses}")
        if questions:
            parts.append(f"### Questions\n{questions}")

        # Follow-up dialogue thread
        follow_ups = reviewer_data.get('dialogue', [])
        if follow_ups:
            parts.append("### Discussion")
            for comment in follow_ups:
                commenter_type = comment.get('commenter_type', 'unknown')
                commenter_id = comment.get('commenter_id', '')
                comment_content = comment.get('content', {})
                text = extract_value(comment_content.get('comment', ''))
                if not text:
                    continue

                if commenter_type == 'author':
                    label = 'Author'
                elif commenter_type == 'reviewer':
                    label = commenter_id
                else:
                    label = commenter_id

                parts.append(f"**{label}**: {text}")

        parts.append("")  # blank line between reviewers

    return "\n\n".join(parts)


def build_splits(input_path: str, output_dir: str, train_ratio: float, val_ratio: float, seed: int):
    with open(input_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    # Format each entry, keeping only action items with non-empty diffs
    formatted = []
    skipped_items = 0
    skipped_papers = 0
    for entry in entries:
        dialogue_str = format_dialogue(entry['dialogues'])
        action_items = [
            {'comment': ai['comment'], 'response': ai['response']}
            for ai in entry['action_items']
            if ai.get('diffs') and len(ai['diffs']) > 0
        ]
        skipped_items += sum(
            1 for ai in entry['action_items']
            if not ai.get('diffs') or len(ai['diffs']) == 0
        )
        if not action_items:
            skipped_papers += 1
            continue
        formatted.append({
            'paper_id': entry['paper_id'],
            'input': dialogue_str,
            'target': {'action_items': action_items},
        })

    print(f"Filtered out {skipped_items} action items with empty diffs")
    print(f"Dropped {skipped_papers} papers with no remaining action items")
    print(f"Kept {len(formatted)} papers with {sum(len(e['target']['action_items']) for e in formatted)} action items")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(formatted)

    n = len(formatted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = formatted[:n_train]
    val = formatted[n_train:n_train + n_val]
    test = formatted[n_train + n_val:]

    # Write splits
    import os
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        path = os.path.join(output_dir, f'{split_name}.jsonl')
        with open(path, 'w') as f:
            for record in split_data:
                f.write(json.dumps(record) + '\n')
        total_items = sum(len(r['target']['action_items']) for r in split_data)
        print(f"{split_name}: {len(split_data)} papers, {total_items} action items -> {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build train/val/test splits for action item detection.")
    parser.add_argument('--input', type=str, default='action_item_dataset_iclr_master.jsonl',
                        help='Input JSONL from build_action_item_dataset.py')
    parser.add_argument('--output-dir', type=str, default='splits',
                        help='Output directory for train/val/test JSONL files')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    build_splits(args.input, args.output_dir, args.train_ratio, args.val_ratio, args.seed)
