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
