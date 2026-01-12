#!/usr/bin/env python3
"""
diff_linking_ui.py

A wrapper script for batch processing reviewer comments through the PDF diff viewer.
For each entry in a JSONL input file, this script:
1. Launches pdf_viewer_clickable.py with the appropriate PDFs and comments
2. Waits for user to interact and save diffs via Shift+Click
3. Collects saved diffs and writes them with the input fields to an output file
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def find_pdf_files(paper_id: str, pdf_dir: str) -> tuple[str | None, str | None]:
    """
    Find PDF files for a given paper_id in the specified directory.
    
    Returns:
        Tuple of (openreview_pdf, arxiv_pdf) paths, or (None, None) if not found
    """
    pdf_dir_path = Path(pdf_dir)
    
    # Look for openreview PDF
    openreview_pdf = pdf_dir_path / f"{paper_id}_openreview.pdf"
    
    # Look for arxiv PDF (pattern: {paper_id}_arxiv_*.pdf)
    arxiv_pdfs = list(pdf_dir_path.glob(f"{paper_id}_arxiv_*.pdf"))
    arxiv_pdf = arxiv_pdfs[0] if arxiv_pdfs else None
    
    openreview_path = str(openreview_pdf) if openreview_pdf.exists() else None
    arxiv_path = str(arxiv_pdf) if arxiv_pdf else None
    
    return openreview_path, arxiv_path


def get_saved_diffs(diff_output_file: str) -> list[dict]:
    """
    Read all diffs from the saved_diffs.jsonl file.
    
    Returns:
        List of diff entries
    """
    diffs = []
    if os.path.exists(diff_output_file):
        with open(diff_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        diffs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return diffs


def clear_saved_diffs(diff_output_file: str):
    """Clear the saved diffs file before processing each entry."""
    if os.path.exists(diff_output_file):
        os.remove(diff_output_file)


def run_pdf_viewer(pdf1: str, pdf2: str | None, reviewer_comment: str, 
                   author_response: str, viewer_script: str,
                   use_cached_diffs: bool = False, save_diff_cache: bool = False,
                   entries_file: str | None = None) -> int:
    """
    Run the PDF viewer with the given parameters.
    
    Args:
        pdf1: Path to first PDF file
        pdf2: Path to second PDF file (optional)
        reviewer_comment: Reviewer comment text to display
        author_response: Author response text to display
        viewer_script: Path to pdf_viewer_clickable.py
        use_cached_diffs: If True, try to use cached diffs instead of recomputing
        save_diff_cache: If True, save computed diffs to cache for reuse
        entries_file: Path to JSON file with list of entries (enables batch mode)
    
    Returns:
        Return code from the viewer process
    """
    cmd = [sys.executable, viewer_script]
    
    # Add PDF files
    cmd.append(pdf1)
    if pdf2:
        cmd.append(pdf2)
    
    # Add reviewer comment and author response (only used if no entries_file)
    if reviewer_comment and not entries_file:
        cmd.extend(['--reviewer-comment', reviewer_comment])
    if author_response and not entries_file:
        cmd.extend(['--author-response', author_response])
    
    # Add cache flags
    if use_cached_diffs:
        cmd.append('--use-cached-diffs')
    if save_diff_cache:
        cmd.append('--save-diff-cache')
    
    # Add entries file for batch mode
    if entries_file:
        cmd.extend(['--entries-file', entries_file])
    
    print(f"Launching PDF viewer...")
    if entries_file:
        print(f"(Batch mode with entries from {entries_file})")
    elif use_cached_diffs:
        print("(Using cached diffs - same paper as previous entry)")
    result = subprocess.run(cmd)
    return result.returncode


def count_output_entries(output_file: str) -> int:
    """
    Count the number of entries already in the output file.
    
    Returns:
        Number of valid JSON entries in the output file
    """
    if not os.path.exists(output_file):
        return 0
    
    count = 0
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    continue
    return count


def group_entries_by_paper_id(entries: list[dict], start_index: int) -> list[tuple[str, list[tuple[int, dict]]]]:
    """
    Group consecutive entries by paper_id starting from start_index.
    
    Returns:
        List of (paper_id, [(original_index, entry), ...]) tuples
    """
    groups = []
    current_paper_id = None
    current_group = []
    
    for i, entry in enumerate(entries[start_index:], start=start_index):
        paper_id = entry.get('paper_id', '')
        
        if paper_id != current_paper_id:
            if current_group:
                groups.append((current_paper_id, current_group))
            current_paper_id = paper_id
            current_group = [(i, entry)]
        else:
            current_group.append((i, entry))
    
    if current_group:
        groups.append((current_paper_id, current_group))
    
    return groups


def process_jsonl_file(input_jsonl: str, pdf_dir: str, output_file: str, 
                       viewer_script: str, diff_output_file: str,
                       start_index: int | None = None):
    """
    Process all entries in the input JSONL file.
    
    Entries with the same paper_id are grouped together to avoid recomputing
    PDF diffs when consecutive entries are from the same paper.
    
    Args:
        input_jsonl: Path to input JSONL file
        pdf_dir: Directory containing PDF files
        output_file: Path to output JSONL file
        viewer_script: Path to pdf_viewer_clickable.py
        diff_output_file: Path to saved_diffs.jsonl
        start_index: Index to start processing from (0-based). If None, auto-detect from output file.
    """
    # Read all entries from input
    entries = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON line: {e}")
                    continue
    
    print(f"Loaded {len(entries)} entries from {input_jsonl}")
    
    # Auto-detect start index from output file if not specified
    if start_index is None:
        start_index = count_output_entries(output_file)
        if start_index > 0:
            print(f"Found {start_index} entries already in output file. Resuming from there.")
    
    print(f"Starting from index {start_index}")
    
    # Group entries by paper_id to avoid recomputing diffs
    paper_groups = group_entries_by_paper_id(entries, start_index)
    print(f"Grouped into {len(paper_groups)} paper group(s)")
    
    # Determine paths for entry queue and results files (same directory as viewer script)
    viewer_dir = Path(viewer_script).parent
    entry_queue_file = str(viewer_dir / "entry_queue.json")
    entry_results_file = str(viewer_dir / "entry_results.jsonl")
    
    entries_processed = 0
    
    # Process each paper group
    for group_idx, (paper_id, group_entries) in enumerate(paper_groups):
        group_size = len(group_entries)
        first_entry_idx = group_entries[0][0]
        
        print(f"\n{'='*60}")
        print(f"Paper group {group_idx + 1}/{len(paper_groups)}: {paper_id}")
        print(f"Contains {group_size} entry/entries (indices {first_entry_idx} to {first_entry_idx + group_size - 1})")
        print(f"{'='*60}")
        
        # Find PDF files for this paper
        openreview_pdf, arxiv_pdf = find_pdf_files(paper_id, pdf_dir)
        
        if not openreview_pdf and not arxiv_pdf:
            print(f"Warning: No PDF files found for paper_id '{paper_id}'. Skipping all {group_size} entries...")
            # Write all entries in this group with empty diffs
            for orig_idx, entry in group_entries:
                output_entry = {**entry, 'diffs': []}
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
                entries_processed += 1
            continue
        
        pdf1 = arxiv_pdf
        pdf2 = openreview_pdf
        
        print(f"PDF 1: {pdf1}")
        print(f"PDF 2: {pdf2 or 'N/A'}")
        
        # Write entry queue file with all entries for this paper group
        queue_entries = [entry for _, entry in group_entries]
        with open(entry_queue_file, 'w', encoding='utf-8') as f:
            json.dump(queue_entries, f, ensure_ascii=False)
        print(f"Wrote {group_size} entries to queue file")
        
        # Clear saved diffs and results before launching viewer
        clear_saved_diffs(diff_output_file)
        if os.path.exists(entry_results_file):
            os.remove(entry_results_file)
        
        # Run the PDF viewer with all entries for this paper
        # The viewer will show "Save & Next Entry" button to cycle through entries
        return_code = run_pdf_viewer(
            pdf1=pdf1,
            pdf2=pdf2,
            reviewer_comment='',  # Will be loaded from entries file
            author_response='',
            viewer_script=viewer_script,
            entries_file=entry_queue_file
        )
        
        print(f"PDF viewer closed with return code: {return_code}")
        
        # Collect results from the entry_results.jsonl file
        results = []
        if os.path.exists(entry_results_file):
            with open(entry_results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        
        print(f"Collected results for {len(results)} entries from viewer")
        
        # Match results back to original entries and write to output
        # Results should be in same order as queue
        for i, (orig_idx, entry) in enumerate(group_entries):
            if i < len(results):
                # Use result from viewer
                output_entry = results[i]
            else:
                # Entry was not processed (viewer closed early)
                output_entry = {**entry, 'diffs': [], 'skipped': True, 'reason': 'viewer_closed_early'}
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
            entries_processed += 1
        
        print(f"Wrote {group_size} entries to output file")
        
        # Clean up queue and results files
        if os.path.exists(entry_queue_file):
            os.remove(entry_queue_file)
        if os.path.exists(entry_results_file):
            os.remove(entry_results_file)
        
        # Calculate remaining groups and prompt to continue
        remaining_groups = len(paper_groups) - group_idx - 1
        total_remaining = sum(len(g[1]) for g in paper_groups[group_idx + 1:])
        
        if total_remaining > 0:
            try:
                prompt_msg = f"\n{total_remaining} entries remaining ({remaining_groups} more paper groups). Continue? [Y/n]: "
                response = input(prompt_msg).strip().lower()
                if response == 'n':
                    print(f"\nStopping. Processed {entries_processed} entries this session.")
                    next_idx = group_entries[-1][0] + 1
                    print(f"Resume later with: --start-index {next_idx}")
                    return
            except (KeyboardInterrupt, EOFError):
                print(f"\n\nInterrupted. Processed {entries_processed} entries this session.")
                next_idx = group_entries[-1][0] + 1
                print(f"Resume later with: --start-index {next_idx}")
                return
    
    print(f"\n{'='*60}")
    print(f"Processing complete! Output saved to: {output_file}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Process JSONL entries through PDF diff viewer for diff linking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python diff_linking_ui.py input.jsonl /path/to/pdfs -o output.jsonl
    
The input JSONL file should have entries with:
    - paper_id: Used to find PDFs ({paper_id}_openreview.pdf, {paper_id}_arxiv_*.pdf)
    - filtered_comment: Reviewer comment to display
    - filtered_response: Author response to display

Use Shift+Click in the PDF viewer to save diffs. When you close the viewer,
the diffs are automatically collected and added to the output.
        """
    )
    
    parser.add_argument(
        'input_jsonl',
        type=str,
        help='Path to input JSONL file with paper entries'
    )
    
    parser.add_argument(
        'pdf_dir',
        type=str,
        help='Directory containing PDF files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output JSONL file (default: input_with_diffs.jsonl)'
    )
    
    parser.add_argument(
        '-s', '--start-index',
        type=int,
        default=None,
        help='Index to start processing from (0-based, default: auto-detect from output file)'
    )
    
    parser.add_argument(
        '--viewer-script',
        type=str,
        default=None,
        help='Path to pdf_viewer_clickable.py (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_jsonl):
        print(f"Error: Input file not found: {args.input_jsonl}")
        sys.exit(1)
    
    # Validate PDF directory
    if not os.path.isdir(args.pdf_dir):
        print(f"Error: PDF directory not found: {args.pdf_dir}")
        sys.exit(1)
    
    # Set default output file
    if args.output is None:
        input_path = Path(args.input_jsonl)
        args.output = str(input_path.parent / f"{input_path.stem}_with_diffs.jsonl")
    
    # Find viewer script
    if args.viewer_script is None:
        # Try to find it relative to this script
        script_dir = Path(__file__).parent
        viewer_candidates = [
            script_dir / "PDF-Diff-Functions" / "pdf_viewer_clickable.py",
            script_dir / "pdf_viewer_clickable.py",
        ]
        for candidate in viewer_candidates:
            if candidate.exists():
                args.viewer_script = str(candidate)
                break
        
        if args.viewer_script is None:
            print("Error: Could not find pdf_viewer_clickable.py. Please specify with --viewer-script")
            sys.exit(1)
    
    if not os.path.exists(args.viewer_script):
        print(f"Error: Viewer script not found: {args.viewer_script}")
        sys.exit(1)
    
    # Determine diff output file location (same directory as viewer script)
    viewer_dir = Path(args.viewer_script).parent
    diff_output_file = str(viewer_dir / "saved_diffs.jsonl")
    
    print(f"Input JSONL: {args.input_jsonl}")
    print(f"PDF Directory: {args.pdf_dir}")
    print(f"Output File: {args.output}")
    print(f"Viewer Script: {args.viewer_script}")
    print(f"Diff Output File: {diff_output_file}")
    if args.start_index is not None:
        print(f"Starting from index: {args.start_index}")
    else:
        print("Starting from index: auto-detect from output file")
    
    # Process the file
    process_jsonl_file(
        input_jsonl=args.input_jsonl,
        pdf_dir=args.pdf_dir,
        output_file=args.output,
        viewer_script=args.viewer_script,
        diff_output_file=diff_output_file,
        start_index=args.start_index
    )


if __name__ == "__main__":
    main()
