# Prepare the dataset by extracting only necessary parts of the dialogue
import json
import logging
import os
from argparse import ArgumentParser
from itertools import product

# Load filtering prompt template once at module level for efficiency
_FILTERING_PROMPT_TEMPLATE = None

def get_filtering_prompt_template() -> str:
    """Load and cache the filtering prompt template from file."""
    global _FILTERING_PROMPT_TEMPLATE
    if _FILTERING_PROMPT_TEMPLATE is None:
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', '0_extract_relevant_info.txt')
        with open(prompt_path, 'r') as f:
            _FILTERING_PROMPT_TEMPLATE = f.read()
    return _FILTERING_PROMPT_TEMPLATE


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
    else:
        review_text = review_content
    
    # Extract dialogue content
    dialogue_content = dialogue.get('dialogue', [])
    if isinstance(dialogue_content, list):
        dialogue_text = json.dumps(dialogue_content, indent=2)
    else:
        dialogue_text = str(dialogue_content)
    
    # Build the input section
    prompt_addition = f"""
## Input

### Review
{review_text}

### Dialogue
{dialogue_text}

### Diff
- **Change Type**: {diff.get('tag', 'unknown')}
- **Original Context**: {diff.get('context_pdf1', '')}
- **Revised Context**: {diff.get('context_pdf2', '')}

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
        dialogues = datum.get('dialogues', [])
        
        # Generate all diff-dialogue pairs
        for diff, dialogue in product(diffs, dialogues):
            prompt = build_prompt(diff, dialogue)
            prepared_data.append({
                'paper_id': paper_id,
                'prompt': prompt,
                'diff': diff,
                'dialogue': dialogue
            })
    
    return prepared_data


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare dataset by extracting dialogues.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSON file containing paper metadata with dialogues.")
    parser.add_argument("--output-file", type=str, help="Path to the output JSON file to save the prepared dataset.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), None))
    logger = logging.getLogger(__name__)
    
    # Load input JSON
    with open(args.input_file, 'r') as f:
        papers_json = json.load(f)
    
    # Prepare dataset
    prepared_json = prepare_dataset(papers_json)
    
    # Save output JSON
    output_file = args.output_file if args.output_file else args.input_file.replace('.json', '_prepared.json')
    with open(output_file, 'w') as f:
        json.dump(prepared_json, f, indent=2)
    
    logger.info(f"Prepared dataset saved to: {output_file}")