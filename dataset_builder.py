"""
ActionItemDatasetBuilder: end-to-end preprocessing pipeline.

Takes merged annotation diffs + dialogue JSON and produces
train/val/test splits ready for action item detection experiments.

Usage as script:
    python dataset_builder.py \
        --merged-diffs iclr_merged_diffs.jsonl \
        --dialogues ICLR.cc/2024/Conference/dialogues/papers_20260313_224048_with_arxiv_with_pdfs_dialogues.json \
        --output-dir splits

Usage as library:
    from dataset_builder import ActionItemDatasetBuilder

    builder = ActionItemDatasetBuilder(
        merged_diffs_path='iclr_merged_diffs.jsonl',
        dialogues_path='path/to/dialogues.json',
    )
    builder.run(output_dir='splits')
"""
import json
import os
import random
from collections import defaultdict


class ActionItemDatasetBuilder:
    """Build action item detection datasets from merged annotations and dialogues."""

    def __init__(
        self,
        merged_diffs_path: str,
        dialogues_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        filter_empty_diffs: bool = True,
    ):
        self.merged_diffs_path = merged_diffs_path
        self.dialogues_path = dialogues_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.filter_empty_diffs = filter_empty_diffs

        self._paper_actions: dict[str, list[dict]] = {}
        self._dialogues: dict = {}
        self._master_dataset: list[dict] = []
        self._formatted: list[dict] = []
        self.splits: dict[str, list[dict]] = {}

    # ── Step 1: Load merged diffs ────────────────────────────────────────────

    def load_merged_diffs(self) -> dict[str, list[dict]]:
        """Load merged diffs JSONL and group action items by paper_id."""
        paper_actions = defaultdict(list)
        with open(self.merged_diffs_path) as f:
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
        self._paper_actions = dict(paper_actions)
        print(f"Loaded {sum(len(v) for v in self._paper_actions.values())} action items "
              f"across {len(self._paper_actions)} papers from {self.merged_diffs_path}")
        return self._paper_actions

    # ── Step 2: Load dialogues ───────────────────────────────────────────────

    def load_dialogues(self) -> dict:
        """Load dialogues JSON keyed by paper ID."""
        with open(self.dialogues_path) as f:
            self._dialogues = json.load(f)
        print(f"Loaded dialogues for {len(self._dialogues)} papers from {self.dialogues_path}")
        return self._dialogues

    # ── Step 3: Build master dataset ─────────────────────────────────────────

    def build_master_dataset(self) -> list[dict]:
        """Join action items with dialogues to create the master dataset.

        Each entry has paper_id, dialogues, and action_items.
        """
        if not self._paper_actions:
            self.load_merged_diffs()
        if not self._dialogues:
            self.load_dialogues()

        dataset = []
        skipped = 0
        for paper_id, action_items in self._paper_actions.items():
            paper_dialogues = self._dialogues.get(paper_id, {}).get('dialogue', [])
            if not paper_dialogues:
                skipped += 1
                continue
            dataset.append({
                'paper_id': paper_id,
                'dialogues': paper_dialogues,
                'action_items': action_items,
            })

        self._master_dataset = dataset
        total_items = sum(len(e['action_items']) for e in dataset)
        print(f"Built master dataset: {len(dataset)} papers, {total_items} action items "
              f"(skipped {skipped} papers with no dialogues)")
        return self._master_dataset

    def save_master_dataset(self, path: str):
        """Write the master dataset to a JSONL file."""
        if not self._master_dataset:
            self.build_master_dataset()
        with open(path, 'w') as f:
            for record in self._master_dataset:
                f.write(json.dumps(record) + '\n')
        print(f"Saved master dataset to {path}")

    # ── Step 4: Format for LLM and filter ────────────────────────────────────

    def format_and_filter(self) -> list[dict]:
        """Format dialogues as strings and filter action items with empty diffs.

        Produces records with:
          - paper_id
          - input: formatted dialogue string
          - target: {action_items: [{comment, response}, ...]}
        """
        if not self._master_dataset:
            self.build_master_dataset()

        formatted = []
        skipped_items = 0
        skipped_papers = 0

        for entry in self._master_dataset:
            dialogue_str = self._format_dialogue(entry['dialogues'])

            if self.filter_empty_diffs:
                action_items = [
                    {'comment': ai['comment'], 'response': ai['response']}
                    for ai in entry['action_items']
                    if ai.get('diffs') and len(ai['diffs']) > 0
                ]
                skipped_items += sum(
                    1 for ai in entry['action_items']
                    if not ai.get('diffs') or len(ai['diffs']) == 0
                )
            else:
                action_items = [
                    {'comment': ai['comment'], 'response': ai['response']}
                    for ai in entry['action_items']
                ]

            if not action_items:
                skipped_papers += 1
                continue

            formatted.append({
                'paper_id': entry['paper_id'],
                'input': dialogue_str,
                'target': {'action_items': action_items},
            })

        self._formatted = formatted
        kept_items = sum(len(e['target']['action_items']) for e in formatted)
        print(f"Formatted {len(formatted)} papers with {kept_items} action items")
        if self.filter_empty_diffs:
            print(f"  Filtered {skipped_items} action items with empty diffs")
            print(f"  Dropped {skipped_papers} papers with no remaining items")
        return self._formatted

    # ── Step 5: Generate splits ──────────────────────────────────────────────

    def generate_splits(self) -> dict[str, list[dict]]:
        """Shuffle and split formatted data into train/val/test."""
        if not self._formatted:
            self.format_and_filter()

        random.seed(self.seed)
        data = list(self._formatted)
        random.shuffle(data)

        n = len(data)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        self.splits = {
            'train': data[:n_train],
            'val': data[n_train:n_train + n_val],
            'test': data[n_train + n_val:],
        }

        for name, split in self.splits.items():
            items = sum(len(r['target']['action_items']) for r in split)
            print(f"  {name}: {len(split)} papers, {items} action items")

        return self.splits

    def save_splits(self, output_dir: str):
        """Write train/val/test splits to JSONL files."""
        if not self.splits:
            self.generate_splits()

        os.makedirs(output_dir, exist_ok=True)
        for name, split in self.splits.items():
            path = os.path.join(output_dir, f'{name}.jsonl')
            with open(path, 'w') as f:
                for record in split:
                    f.write(json.dumps(record) + '\n')
            items = sum(len(r['target']['action_items']) for r in split)
            print(f"  {name}: {len(split)} papers, {items} action items -> {path}")

    # ── Full pipeline ────────────────────────────────────────────────────────

    def run(self, output_dir: str, master_path: str | None = None):
        """Run the full pipeline: load -> join -> filter -> split -> save."""
        self.load_merged_diffs()
        self.load_dialogues()
        self.build_master_dataset()
        if master_path:
            self.save_master_dataset(master_path)
        self.format_and_filter()
        self.generate_splits()
        self.save_splits(output_dir)

    # ── Dialogue formatting ──────────────────────────────────────────────────

    @staticmethod
    def _extract_value(field) -> str:
        """Extract text value from OpenReview content fields."""
        if isinstance(field, dict):
            return field.get('value', '')
        return str(field) if field else ''

    @classmethod
    def _format_dialogue(cls, dialogues: list[dict]) -> str:
        """Format reviewer dialogues into a single readable string."""
        parts = []

        for reviewer_data in dialogues:
            reviewer_id = reviewer_data.get('reviewer_id', 'Unknown')
            review = reviewer_data.get('review', {})
            content = review.get('content', {})

            weaknesses = cls._extract_value(content.get('weaknesses', ''))
            questions = cls._extract_value(content.get('questions', ''))

            parts.append(f"## {reviewer_id}")

            if weaknesses:
                parts.append(f"### Weaknesses\n{weaknesses}")
            if questions:
                parts.append(f"### Questions\n{questions}")

            follow_ups = reviewer_data.get('dialogue', [])
            if follow_ups:
                parts.append("### Discussion")
                for comment in follow_ups:
                    commenter_type = comment.get('commenter_type', 'unknown')
                    commenter_id = comment.get('commenter_id', '')
                    comment_content = comment.get('content', {})
                    text = cls._extract_value(comment_content.get('comment', ''))
                    if not text:
                        continue

                    if commenter_type == 'author':
                        label = 'Author'
                    else:
                        label = commenter_id

                    parts.append(f"**{label}**: {text}")

            parts.append("")

        return "\n\n".join(parts)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Build action item detection dataset and splits.")
    parser.add_argument('--merged-diffs', type=str, default='iclr_merged_diffs.jsonl')
    parser.add_argument('--dialogues', type=str,
                        default='ICLR.cc/2024/Conference/dialogues/papers_20260313_224048_with_arxiv_with_pdfs_dialogues.json')
    parser.add_argument('--output-dir', type=str, default='splits')
    parser.add_argument('--master-output', type=str, default=None,
                        help='Optional path to save master dataset JSONL')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-filter-empty-diffs', action='store_true')
    args = parser.parse_args()

    builder = ActionItemDatasetBuilder(
        merged_diffs_path=args.merged_diffs,
        dialogues_path=args.dialogues,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        filter_empty_diffs=not args.no_filter_empty_diffs,
    )
    builder.run(output_dir=args.output_dir, master_path=args.master_output)
