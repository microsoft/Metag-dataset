### Install packages
sudo apt install python3-tk
sudo apt-get install idle3



### Set up virtual environment  
```
uv venv review-env --python 3.12.3
source review-env/bin/activate
uv pip install -r requirements.txt
```


### Environment Variables to set:

#### OpenReview
export OPENREVIEW_USERNAME=<openreview-username>
export OPENREVIEW_PASSWORD=<openreview-password>


#### Semantic Scholar
Request a key: https://www.semanticscholar.org/product/api#api-key 
export SEMANTICSCHOLAR_API_KEY=<ss-api-key>

### Execution
```
python scraper.py
python dialogue_diff.py
python data_preparer_filtering.py --input-file ./ICLR.cc/2024/Conference/dialogues/papers_20251208_225455_with_arxiv_with_pdfs_dialogues.json --output-file extracted_dialogue_pairs.json --expt review_action_items
```

### UI elements for human annotation
```
python comment_filtering_ui.py
python diff_linking_ui.py ./extracted_dialogue_pairs_filtered.jsonl ./ICLR.cc/2024/Conference/PDF2/ -o tmp_outputs_diff_linked.jsonl
```

Comment filtering allows to keep/discard a comment-response pair
The clickable PDF viewer allows to view a comment-response pair and click on the corresponding diffs to save them


### Install PyTorch
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```


### Install vLLM
Use docker to run vLLM: 
```
docker pull vllm/vllm-openai:latest
```

Then, the inference code will send prompts to vLLM using:
```
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model <model-name>
```


### Action Item Detection Pipeline

#### 1. Merge annotations from multiple annotators
```
python merge_annotations.py --input-dir annotated_data --output-file annotated_data/merged_annotations.jsonl
```

#### 2. Build dataset and train/val/test splits
```
python dataset_builder.py \
    --merged-diffs iclr_merged_diffs.jsonl \
    --dialogues ICLR.cc/2024/Conference/dialogues/papers_20260313_224048_with_arxiv_with_pdfs_dialogues.json \
    --output-dir splits \
    --master-output action_item_dataset_iclr_master.jsonl
```

Or use as a library:
```python
from dataset_builder import ActionItemDatasetBuilder

builder = ActionItemDatasetBuilder(
    merged_diffs_path='iclr_merged_diffs.jsonl',
    dialogues_path='path/to/dialogues.json',
)
builder.run(output_dir='splits', master_path='master.jsonl')
```

#### 3. Run zero-shot baseline
```
python run_action_item_experiment.py --mode zero-shot --split test
```
If the vLLM server is already running:
```
python run_action_item_experiment.py --mode zero-shot --split test --no-auto-start
```

#### 4. Run fine-tuning experiment
```
python run_action_item_experiment.py --mode finetune --split test \
    --epochs 3 --lr 2e-5 --lora-r 8
```

#### 5. Evaluate saved results
```python
from evaluator import ActionItemEvaluator
import json

results = [json.loads(line) for line in open('experiment_results/results.jsonl')]
evaluator = ActionItemEvaluator()
metrics = evaluator.evaluate_all(results)
ActionItemEvaluator.print_summary(metrics)
```

---

### Diff Prediction Pipeline

Given an action item and all the PDF diffs for a paper, predict which specific
diffs correspond to the change described by the action item.

#### Prerequisites

You need three things before building the diff prediction dataset:

1. **Master action-item dataset** (`action_item_dataset_iclr_master.jsonl`) —
   contains papers with dialogues and action items, where each action item has
   annotator-linked diffs.
2. **Full paper diffs JSON** — the output of the PDF diff extractor
   (e.g. `ICLR.cc/2024/Conference/diffs/papers_*_with_arxiv_with_pdfs_diffs.json`).
3. **Existing action-item splits** (`splits/`) — used to assign papers to
   train/val/test so there is no paper leakage across sets.

#### Build the dataset (ICLR)

```bash
python build_diff_prediction_dataset.py --strip-word-positions
```

This uses the defaults:
- `--master action_item_dataset_iclr_master.jsonl`
- `--diffs ICLR.cc/2024/Conference/diffs/papers_20260313_224048_with_arxiv_with_pdfs_diffs.json`
- `--splits-dir splits`
- `--output-dir diff_prediction_splits`

Output: `diff_prediction_splits/{train,val,test}.jsonl`

#### Build the dataset (NeurIPS)

Once you have the NeurIPS master dataset and action-item splits, point the
arguments at the NeurIPS paths:

```bash
python build_diff_prediction_dataset.py \
    --master action_item_dataset_neurips_master.jsonl \
    --diffs NeurIPS.cc/2024/Conference/diffs/papers_20260312_222512_with_arxiv_with_pdfs_diffs.json \
    --splits-dir neurips_splits \
    --output-dir diff_prediction_splits_neurips \
    --strip-word-positions
```

#### Output schema

Each line in the output JSONL is one (action_item, paper) pair:

| Field | Description |
|---|---|
| `paper_id` | Paper identifier |
| `dialogue` | Formatted reviewer-author dialogue |
| `action_item` | `{comment, response}` — the single action item |
| `all_diffs` | All PDF diffs for the paper. Each diff has a `diff_index` field (0-based position in the paper's full diff list) |
| `correct_diff_indices` | List of indices into `all_diffs` that are the ground-truth diffs for this action item |

To look up a correct diff: `entry['all_diffs'][idx]` where `idx` is in
`correct_diff_indices`. The `diff_index` field on each diff matches its
position so you can verify: `entry['all_diffs'][idx]['diff_index'] == idx`.

#### How diff matching works

Annotated diffs (from the annotation UI) and full paper diffs (from the PDF
diff extractor) have different schemas. The matching algorithm reconciles them:

1. The annotated diff's `pane` field (`"right"` = revised PDF, `"left"` =
   original) determines which side of the paper diff to compare against
   (`text_pdf2`/`context_*_pdf2` vs `text_pdf1`/`context_*_pdf1`).

2. A scoring system ranks each candidate paper diff:
   - **+10** if annotated `diff_text` is a substring of paper diff text (or +8 vice versa)
   - **+3** each for matching `context_before` and `context_after`
   - **+2** for page number match
   - **+1** for diff type match (`insertion`→`insert`/`replace`, etc.)

3. A minimum score of 10 (requiring at least a text match) is needed to accept.

This achieves a ~97.5% match rate (2686/2754 individual annotated diffs).

#### Options

| Flag | Default | Description |
|---|---|---|
| `--master` | `action_item_dataset_iclr_master.jsonl` | Master dataset JSONL |
| `--diffs` | `ICLR.cc/.../diffs/papers_*_diffs.json` | Full paper diffs JSON |
| `--splits-dir` | `splits` | Existing action-item splits dir |
| `--output-dir` | `diff_prediction_splits` | Output directory |
| `--strip-word-positions` | off | Remove word bounding boxes (smaller files) |
| `--seed` | `42` | Random seed for shuffling |

---

### Diff Classification Experiment

Given an action item and the full set of PDF diffs for a paper, classify which
diffs are relevant to the change described by the action item.

**Approach:** Diffs are sent to an LLM in batches (default 50 per prompt). For
each batch the LLM returns the `diff_index` values of relevant diffs. Trivial
formatting diffs (≤2 words) are pre-filtered by default to reduce prompt count.

Evaluation uses precision, recall, and F1 (micro and macro) against the
ground-truth `correct_diff_indices`.

#### 1. Zero-shot
```bash
python run_diff_classification_experiment.py --mode zero-shot --split test
```

Against an already-running vLLM server:
```bash
python run_diff_classification_experiment.py --mode zero-shot --split test --no-auto-start
```

Adjust batch size of diffs per prompt:
```bash
python run_diff_classification_experiment.py --mode zero-shot --split test --diffs-per-prompt 100
```

#### 2. Fine-tune on train, evaluate on test
```bash
python run_diff_classification_experiment.py --mode finetune --split test
```

With custom hyperparams:
```bash
python run_diff_classification_experiment.py --mode finetune --split test \
    --epochs 5 --lr 2e-5 --lora-r 16
```

#### Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | `zero-shot` | `zero-shot` or `finetune` |
| `--split` | `test` | Which split to evaluate on |
| `--splits-dir` | `diff_prediction_splits` | Directory with train/val/test JSONL |
| `--diffs-per-prompt` | `50` | Number of diffs per LLM call |
| `--no-prefilter` | off | Disable trivial-diff pre-filtering |
| `--config` | `config.yaml` | Inference config YAML |
| `--output-dir` | `experiment_results` | Output directory |
| `--no-auto-start` | off | Skip auto-starting vLLM Docker |
| `--use-vllm-inference` | off | Use vLLM (not in-memory) after fine-tuning |
| `--epochs` | `3` | Fine-tuning epochs |
| `--lr` | `2e-5` | Learning rate |
| `--lora-r` | `8` | LoRA rank |
