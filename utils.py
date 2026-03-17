import json
import re

def parse_json_response(response_string: str) -> dict:
    """Extract and parse JSON from a response string."""
    # Try to find JSON between code fences
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_string)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Assume the whole string is JSON
        json_str = response_string.strip()
    
    return json.loads(json_str)


def load_jsonl(file_path: str) -> list[dict]:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data