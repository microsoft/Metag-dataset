# Metag Dataset

Metag links reviewer-derived action items to the revisions that implement them
in scientific papers. Each example contains a reviewer concern, the corresponding
author response, the candidate PDF diffs between the original and revised paper,
and labels identifying the relevant diffs.

## Dataset

The release is under `data/diff_classification/`:

| File | Description |
|---|---|
| `train.jsonl` | One full diff-classification example per action item. |
| `val.jsonl` | Validation examples with complete candidate diff pools. |
| `test.jsonl` | Test examples with complete candidate diff pools. |
| `train_windows.jsonl` | Model-facing training prompts containing up to 40 diffs. |
| `val_windows.jsonl` | Model-facing validation prompts covering every candidate diff. |
| `test_windows.jsonl` | Model-facing test prompts covering every candidate diff. |
| `stats.json` | Dataset configuration and split statistics. |

### Statistics

| Split | Papers | Action items | Candidate diffs | Relevant diffs | Windows |
|---|---:|---:|---:|---:|---:|
| Train | 93 | 248 | 219,569 | 1,557 | 999 |
| Validation | 22 | 41 | 41,445 | 273 | 1,060 |
| Test | 22 | 60 | 62,772 | 494 | 1,599 |
| **Total** | **137** | **349** | **323,786** | **2,324** | **3,658** |

There is no paper overlap between splits. An action item has 927.8 candidate
diffs and 6.66 relevant diffs on average.

## Example Schema

Each line in `train.jsonl`, `val.jsonl`, and `test.jsonl` is a JSON object:

```json
{
    "example_id": "paper_id::ai0",
    "paper_id": "paper_id",
    "split": "test",
    "action_item": {
        "comment": "Reviewer concern",
        "response": "Author description of the revision"
    },
    "all_diffs": [
        {
            "diff_index": 0,
            "tag": "replace",
            "is_moved": false,
            "page": 0,
            "old": "Text in the original paper",
            "new": "Text in the revised paper",
            "context_before": "Preceding text",
            "context_after": "Following text"
        }
    ],
    "relevant_diff_indices": [0],
    "labels": [true],
    "n_diffs": 1,
    "n_relevant": 1
}
```

`diff_index` is global within a paper. The Boolean values in `labels` are aligned
with `all_diffs`, while `relevant_diff_indices` provides the positive indices
directly.

## Prompt Windows

Papers often contain more candidate diffs than fit in one model prompt. The
window files divide each action item's candidate pool into contiguous groups of
up to 40 diffs and include the rendered prompt and JSON target. Validation and
test windows cover every candidate diff. Training windows use negative
subsampling: all positive windows are retained, together with up to two
all-negative windows per positive window and at least two negative windows per
example.

## Intended Use

Metag is released for research on action-item extraction, scientific-document
revision analysis, and diff classification. It should not be used to evaluate
paper quality, predict review outcomes, or make publication or other high-risk
decisions without substantial additional validation.

> Trademarks: This project may contain trademarks or logos for projects,
> products, or services. Authorized use of Microsoft trademarks or logos is
> subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of
> Microsoft trademarks or logos in modified versions of this project must not
> cause confusion or imply Microsoft sponsorship. Any use of third-party
> trademarks or logos is subject to those third parties' policies.


