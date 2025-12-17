# Prepare the dataset by extracting only necessary parts of the dialogue
import json
import logging
import os
from argparse import ArgumentParser
from itertools import product
from inference import InferenceConfig, HuggingFaceInference, VLLMInference, UnslothInference
from tqdm import tqdm
import utils 

# Load filtering prompt template once at module level for efficiency
_FILTERING_PROMPT_TEMPLATE = None

# Load a prompt for filtering action items from reviews
_REVIEW_ACTION_ITEMS_PROMPT = None

def get_filtering_prompt_template() -> str:
    """Load and cache the filtering prompt template from file."""
    global _FILTERING_PROMPT_TEMPLATE
    if _FILTERING_PROMPT_TEMPLATE is None:
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', '0_extract_relevant_info.txt')
        with open(prompt_path, 'r') as f:
            _FILTERING_PROMPT_TEMPLATE = f.read()
    return _FILTERING_PROMPT_TEMPLATE


def get_review_action_items_prompt() -> str:
    """Load and cache the review action items prompt from file."""
    global _REVIEW_ACTION_ITEMS_PROMPT
    if _REVIEW_ACTION_ITEMS_PROMPT is None:
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', '1_extract_review_action_items.txt')
        with open(prompt_path, 'r') as f:
            _REVIEW_ACTION_ITEMS_PROMPT = f.read()
    return _REVIEW_ACTION_ITEMS_PROMPT


def build_prompt(diff: dict, dialogue: dict) -> str:
    """
    Build a complete prompt by combining the base prompt template with the diff and dialogue data.
    
    Args:
        diff: Dictionary containing diff information with keys:
            - tag: The type of change (insert, delete, replace)
            - context_pdf1: The original text with surrounding context
            - context_pdf2: The revised text with surrounding context
        dialogue: Dictionary containing dialogue information with keys:
            - review: The original review content
            - dialogue: The back-and-forth dialogue
    
    Returns:
        Complete prompt string ready for LLM inference
    """
    base_prompt = get_filtering_prompt_template()
    
    # Extract review content - handle nested structure
    review_content = dialogue.get('review', {})
    if isinstance(review_content, dict):
        review_text = review_content.get('content', review_content)
        review_text_relevant = review_text.get('weaknesses', '').get('value','') + "\n" + review_text.get('questions', '').get('value','')
    else:
        review_text = review_content
        
    
    # Extract dialogue content
    dialogue_content = dialogue.get('dialogue', [])

    if isinstance(dialogue_content, list):
        dialogue_parts = []
        for item in dialogue_content:
            content = item.get('content', {})
            comment_value = content.get('comment', {}).get('value', '') if isinstance(content, dict) else ''
            dialogue_parts.append(comment_value)
        dialogue_text = "\n".join(dialogue_parts)
    else:
        dialogue_text = str(dialogue_content)
    
    # Build the input section
    prompt_addition = f"""
## Input

### Review
{review_text_relevant}

### Dialogue
{dialogue_text}

### Diff
- **Change Type**: {diff.get('tag', 'unknown')}
- **Original Text**: {diff.get('text_pdf1', '')}
- **Revised Text**: {diff.get('text_pdf2', '')}
- **Original Context**: {diff.get('context_pdf1', '')}
- **Revised Context**: {diff.get('context_pdf2', '')}

## Output"""

    return base_prompt + prompt_addition


def build_action_items_prompt(dialogue: dict) -> str:
    """
    Build a prompt to extract action items from the review dialogue.
    
    Args:
        dialogue: Dictionary containing dialogue information with keys:
            - review: The original review content
            - dialogue: The back-and-forth dialogue
    
    Returns:
        Complete prompt string ready for LLM inference
    """
    base_prompt = get_review_action_items_prompt()
    
    # Extract review content - handle nested structure
    review_content = dialogue.get('review', {})
    if isinstance(review_content, dict):
        review_text = review_content.get('content', review_content)
        review_text_relevant = review_text.get('weaknesses', '').get('value','') + "\n" + review_text.get('questions', '').get('value','')
    else:
        review_text = review_content

    # Extract dialogue content
    dialogue_content = dialogue.get('dialogue', [])

    if isinstance(dialogue_content, list):
        dialogue_parts = []
        for item in dialogue_content:
            content = item.get('content', {})
            comment_value = content.get('comment', {}).get('value', '') if isinstance(content, dict) else ''
            dialogue_parts.append(comment_value)
        dialogue_text = "\n".join(dialogue_parts)
    else:
        dialogue_text = str(dialogue_content)
    
    # Build the input section
    prompt_addition = f"""
    ## Input

    ### Review
    {review_text_relevant}

    ### Dialogue
    {dialogue_text}

    ## Output"""

    return base_prompt + prompt_addition


def prepare_dataset(papers_json: list[dict]) -> list[dict]:
    """
    Prepare the dataset by splitting each datum into an opcode-dialogue pair
    """
    # Pre-load the filtering prompt template once
    get_filtering_prompt_template()
    
    prepared_data = []
    for datum in papers_json:
        paper_id = datum.get('id', 'unknown')
        diffs = datum.get('diffs', [])
        dialogues = datum.get('dialogue', [])
        
        # Generate all diff-dialogue pairs
        for diff, dialogue in product(diffs, dialogues):
            prompt = build_prompt(diff, dialogue)
            prepared_data.append({
                'paper_id': paper_id,
                'prompt': prompt,
            })
            
    return prepared_data


def prepare_review_action_items_dataset(papers_json: list[dict]) -> list[dict]:
    """
    Prepare the dataset by splitting each datum into an opcode-dialogue pair
    """
    # Pre-load the filtering prompt template once
    get_review_action_items_prompt()
    
    prepared_data = []
    for datum in papers_json:
        paper_id = datum.get('id', 'unknown')
        dialogues = datum.get('dialogue', [])
        for dialogue in dialogues:
            prepared_data.append({
                'paper_id': paper_id,
                'prompt': build_action_items_prompt(dialogue),
            })
        
    return prepared_data


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare dataset by extracting dialogues.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSON file containing paper metadata with dialogues.")
    parser.add_argument("--output-file", type=str, help="Path to the output JSON file to save the prepared dataset.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--expt", type=str, choices=["filtering", "review_action_items"], default="review_action_items", help="Type of experiment to prepare dataset for.")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), None))
    logger = logging.getLogger(__name__)
    
    # Load input JSON
    with open(args.input_file, 'r') as f:
        papers_json = json.load(f)
    
    # Prepare dataset
    if args.expt == "filtering":
        prepared_data = prepare_dataset(papers_json)
    elif args.expt == "review_action_items":
        prepared_data = prepare_review_action_items_dataset(papers_json)
    else:
        raise NotImplementedError(f"Unknown experiment type: {args.expt}")

    config = InferenceConfig.from_yaml("config.yaml")

    prompts = [item['prompt'] for item in prepared_data]
    
    # Use VLLMInference with Docker-based server
    # Set auto_start_server=False if vLLM server is already running
    with VLLMInference(config, auto_start_server=True) as inference:
        responses = inference.generate(prompts, batch_size=config.batch_size)
        
        for i, response in enumerate(responses):
            try:
                json_response = utils.parse_json_response(response)
                prepared_data[i]['response'] = json_response
            except Exception as e:
                logger.error(f"Error parsing response for prompt {i}: {e}")
                prepared_data[i]['response'] = ''
        
    import bpdb; bpdb.set_trace()
    
    # Save output JSON
    output_file = args.output_file if args.output_file else args.input_file.replace('.json', '_prepared.json')
    with open(output_file, 'w') as f:
        json.dump(prepared_data, f, indent=2)
    
    logger.info(f"Prepared dataset saved to: {output_file}")