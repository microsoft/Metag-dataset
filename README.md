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


> Trademarks This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.


