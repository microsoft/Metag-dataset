### Install packages
sudo apt install python3-tk


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
python data_preparer_filtering.py --expt review_action_items
python comment_filtering_ui.py
```


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


