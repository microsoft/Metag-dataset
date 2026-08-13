"""Validate the released Metag diff-classification files."""

import json
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data" / "diff_classification"


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    message = f"{path}:{line_number}: invalid JSON"
                    raise ValueError(message) from error


def validate_split(split: str, expected: dict) -> None:
    examples = list(load_jsonl(DATA_DIR / f"{split}.jsonl"))
    windows = list(load_jsonl(DATA_DIR / f"{split}_windows.jsonl"))

    assert len(examples) == expected["examples"]
    paper_ids = {example["paper_id"] for example in examples}
    assert len(paper_ids) == expected["papers"]
    assert len(windows) == expected["windows"]["total"]

    for example in examples:
        diffs = example["all_diffs"]
        relevant = set(example["relevant_diff_indices"])
        diff_indices = {diff["diff_index"] for diff in diffs}

        assert example["n_diffs"] == len(diffs)
        assert example["n_relevant"] == len(relevant)
        assert relevant <= diff_indices
        assert example["labels"] == [
            diff["diff_index"] in relevant for diff in diffs
        ]


def main() -> None:
    with (DATA_DIR / "stats.json").open(encoding="utf-8") as handle:
        stats = json.load(handle)["splits"]

    for split in ("train", "val", "test"):
        validate_split(split, stats[split])

    print("Metag dataset validation passed.")


if __name__ == "__main__":
    main()
