"""
Merge annotations from multiple annotators.
For samples annotated by two people, compute the intersection of their diffs
and write the merged results to a new JSONL file.
"""
import json
import os
import argparse
from collections import defaultdict


def load_annotator_data(annotated_dir: str) -> dict[str, dict[str, dict]]:
    """Load all annotator JSONL files, keyed by annotator name -> unique_id -> entry."""
    annotators = {}
    for f in sorted(os.listdir(annotated_dir)):
        if not f.endswith('.jsonl'):
            continue
        name = f.split('_outputs')[0]
        entries = {}
        with open(os.path.join(annotated_dir, f)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                entries[d['unique_id']] = d
        annotators[name] = entries
        print(f"Loaded {len(entries)} entries from {name}")
    return annotators


def diff_key(diff: dict) -> tuple:
    """Create a hashable key for a diff to enable set intersection."""
    return (
        diff.get('pane'),
        diff.get('file_name'),
        diff.get('diff_type'),
        diff.get('page_num'),
        diff.get('diff_text'),
        diff.get('context_before'),
        diff.get('context_after'),
    )


def intersect_diffs(diffs_a: list[dict], diffs_b: list[dict]) -> list[dict]:
    """Return diffs present in both annotators' lists."""
    keys_b = {diff_key(d) for d in diffs_b}
    return [d for d in diffs_a if diff_key(d) in keys_b]


def merge_annotations(annotated_dir: str, output_file: str):
    annotators = load_annotator_data(annotated_dir)

    # Map unique_id -> list of (annotator_name, entry)
    uid_map = defaultdict(list)
    for name, entries in annotators.items():
        for uid, entry in entries.items():
            uid_map[uid].append((name, entry))

    paired = {uid: items for uid, items in uid_map.items() if len(items) == 2}
    unpaired = {uid: items for uid, items in uid_map.items() if len(items) == 1}

    print(f"\nTotal unique IDs: {len(uid_map)}")
    print(f"Paired (2 annotators): {len(paired)}")
    print(f"Unpaired (1 annotator): {len(unpaired)}")

    merged = []
    for uid, items in paired.items():
        (name_a, entry_a), (name_b, entry_b) = items

        diffs_a = entry_a.get('diffs', [])
        diffs_b = entry_b.get('diffs', [])
        common_diffs = intersect_diffs(diffs_a, diffs_b)

        # Use entry_a as base and replace diffs with the intersection
        merged_entry = dict(entry_a)
        merged_entry['diffs'] = common_diffs
        merged_entry['annotators'] = [name_a, name_b]
        merged_entry['diffs_count_a'] = len(diffs_a)
        merged_entry['diffs_count_b'] = len(diffs_b)
        merged_entry['diffs_count_intersection'] = len(common_diffs)
        merged.append(merged_entry)

    # Write output
    with open(output_file, 'w') as f:
        for entry in merged:
            f.write(json.dumps(entry) + '\n')

    print(f"\nWrote {len(merged)} merged entries to {output_file}")

    # Summary stats
    total_a = sum(e['diffs_count_a'] for e in merged)
    total_b = sum(e['diffs_count_b'] for e in merged)
    total_inter = sum(e['diffs_count_intersection'] for e in merged)
    print(f"Total diffs annotator A: {total_a}")
    print(f"Total diffs annotator B: {total_b}")
    print(f"Total intersection diffs: {total_inter}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge paired annotations by intersecting diffs.")
    parser.add_argument('--input-dir', type=str, default='annotated_data',
                        help='Directory containing annotator JSONL files')
    parser.add_argument('--output-file', type=str, default='merged_annotations.jsonl',
                        help='Output JSONL file for merged data')
    args = parser.parse_args()

    merge_annotations(args.input_dir, args.output_file)
