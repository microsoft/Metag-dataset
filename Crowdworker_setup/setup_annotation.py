# Load the jsonl file and divide entries up into batches for annotators to receive
import json
import os
from typing import List, Dict, Tuple
import uuid


ANNOTATORS_LIST = ["aidanjay", "lalice", "anisundar", "dtadimeti","gemmazhang", "chemin", "nicholasyi","nkumankumah","pravva", "prmahey", "sadidhasan", "sivasudev", "sharmasomya", "wajagbawa", "waxiao", "yuekang"]


def load_jsonl(file_path: str) -> List[Dict]:
    """Load a JSONL file and return a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


def add_unique_ids(data: List[Dict]) -> List[Dict]:
    """Add unique identifiers to each entry for tracking."""
    for i, entry in enumerate(data):
        entry['unique_id'] = f"entry_{i:05d}_{uuid.uuid4().hex[:8]}"
    return data


def divide_into_batches(data: List[Dict], num_annotators: int, num_repeats: int = 2) -> Dict[str, List[Dict]]:
    """
    Divide data into batches for annotators such that:
    1. Each entry is annotated by exactly num_repeats different annotators
    2. Each annotator gets roughly equal number of entries
    3. Order of entries is preserved (no shuffling)
    
    Strategy: Divide data into num_annotators chunks. Each chunk is assigned to 
    num_repeats consecutive annotators (with wraparound).
    """
    # Initialize batches for each annotator
    batches = {annotator: [] for annotator in ANNOTATORS_LIST[:num_annotators]}
    annotator_names = ANNOTATORS_LIST[:num_annotators]
    
    # Calculate chunk size - divide total entries by number of annotators
    total_entries = len(data)
    chunk_size = total_entries // num_annotators
    remainder = total_entries % num_annotators
    
    # Create chunks while preserving order
    chunks = []
    start_idx = 0
    for i in range(num_annotators):
        # Distribute remainder across first few chunks
        extra = 1 if i < remainder else 0
        end_idx = start_idx + chunk_size + extra
        chunks.append(data[start_idx:end_idx])
        start_idx = end_idx
    
    # Assign each chunk to num_repeats consecutive annotators
    for chunk_idx, chunk in enumerate(chunks):
        for repeat in range(num_repeats):
            annotator_idx = (chunk_idx + repeat) % num_annotators
            annotator_name = annotator_names[annotator_idx]
            batches[annotator_name].extend(chunk)
    
    return batches


def create_annotator_files(input_file: str, output_dir: str, num_repeats: int = 2):
    """
    Main function to create annotator-specific JSONL files.
    
    Args:
        input_file: Path to the input JSONL file
        output_dir: Directory to save annotator files
        num_repeats: Number of annotators per entry (default 2)
    """
    # Load data
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} entries from {input_file}")
    
    # Add unique identifiers
    data = add_unique_ids(data)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the master file with unique IDs for later stitching
    master_file = os.path.join(output_dir, "master_with_ids.jsonl")
    save_jsonl(data, master_file)
    print(f"Saved master file with unique IDs to {master_file}")
    
    # Divide into batches
    num_annotators = len(ANNOTATORS_LIST)
    batches = divide_into_batches(data, num_annotators, num_repeats)
    

    
    # Save each annotator's batch
    for annotator, entries in batches.items():
        if entries:  # Only create file if there are entries
            output_file = os.path.join(output_dir, f"{annotator}_batch.jsonl")
            save_jsonl(entries, output_file)
            print(f"Saved {len(entries)} entries for {annotator} to {output_file}")
    
    # Print summary
    print("\n--- Summary ---")
    print(f"Total entries: {len(data)}")
    print(f"Number of annotators: {num_annotators}")
    print(f"Repeats per entry: {num_repeats}")
    print(f"Expected total annotations: {len(data) * num_repeats}")
    actual_total = sum(len(entries) for entries in batches.values())
    print(f"Actual total assignments: {actual_total}")
    
    # Print per-annotator stats
    print("\nPer-annotator breakdown:")
    for annotator in ANNOTATORS_LIST:
        count = len(batches.get(annotator, []))
        print(f"  {annotator}: {count} entries")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Divide JSONL data into annotator batches")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("--output_dir", default="annotator_batches", help="Output directory for annotator files")
    parser.add_argument("--num_repeats", type=int, default=2, help="Number of annotators per entry")
    
    args = parser.parse_args()
    
    create_annotator_files(args.input_file, args.output_dir, args.num_repeats)