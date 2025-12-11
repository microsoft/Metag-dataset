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
python data_preparer.py
```

