#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
PDF Diff Extractor - Extract word-level differences between two PDFs and save to JSON

This script extracts words from two PDF files, compares them using difflib.SequenceMatcher,
and outputs the opcodes with corresponding text content to a JSON file.
"""

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import fitz  # PyMuPDF


def extract_words_from_pdf(pdf_path, ignore_ligatures=False):
    """
    Extracts all words from a PDF document in reading order.
    
    Args:
        pdf_path (str): Path to the PDF file
        ignore_ligatures (bool): If True, don't expand ligatures
    
    Returns:
        list: List of word strings in reading order
    """
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF '{pdf_path}': {e}", file=sys.stderr)
        return None
    
    all_words = []
    LINE_TOLERANCE_Y = 3  # Tolerance for grouping words into lines
    
    for page_num, page in enumerate(pdf_document):
        page.remove_rotation()
        
        # Extract words with coordinates
        if ignore_ligatures:
            words_data = page.get_text("words", flags=0)
        else:
            words_data = page.get_text("words")
        
        # Group words by block and line for proper reading order
        top_left_in_block = dict()
        grouped_lines = []
        
        for word_info in words_data:
            x0, y0, x1, y1, word_text, block_no, _, _ = word_info[:8]
            word_center_y = (y0 + y1) / 2
            added_to_existing_line = False
            
            # Track top-left corner of each block
            if block_no not in top_left_in_block:
                top_left_in_block[block_no] = (x0, y0)
            else:
                if y0 < top_left_in_block[block_no][1] or \
                   (y0 == top_left_in_block[block_no][1] and x0 < top_left_in_block[block_no][0]):
                    top_left_in_block[block_no] = (x0, y0)
            
            # Group words into lines
            for line_group in grouped_lines:
                if abs(line_group['y_center'] - word_center_y) < LINE_TOLERANCE_Y and \
                   line_group['block_no'] == block_no:
                    line_group['words'].append(word_info)
                    line_group['y_center'] = sum((w[1] + w[3]) / 2 for w in line_group['words']) / len(line_group['words'])
                    added_to_existing_line = True
                    break
            
            if not added_to_existing_line:
                grouped_lines.append({
                    'y_center': word_center_y,
                    'words': [word_info],
                    'block_no': block_no
                })
        
        # Sort lines by block position, then by y-coordinate
        grouped_lines.sort(key=lambda lg: (
            top_left_in_block[lg['block_no']][1],
            top_left_in_block[lg['block_no']][0],
            lg['y_center']
        ))
        
        # Extract words in reading order
        for line_group in grouped_lines:
            line_group['words'].sort(key=lambda w: w[0])  # Sort by x0 coordinate
            for word_info in line_group['words']:
                word_text = word_info[4]
                all_words.append(word_text)
    
    pdf_document.close()
    return all_words


def normalize_words(words, case_insensitive=False, ignore_quotes=False):
    """
    Normalize words for comparison.
    
    Args:
        words (list): List of word strings
        case_insensitive (bool): Convert to lowercase
        ignore_quotes (bool): Normalize different quote types
    
    Returns:
        list: Normalized word list
    """
    normalized = words.copy()
    
    if case_insensitive:
        normalized = [word.lower() for word in normalized]
    
    if ignore_quotes:
        normalized = [
            word.replace("'", "'").replace("'", "'").replace("ʼ", "'")
                .replace('"', '"').replace('"', '"')
            for word in normalized
        ]
    
    return normalized


def get_text_from_indices(words, start_idx, end_idx):
    """
    Get text from a word list given indices.
    
    Args:
        words (list): List of words
        start_idx (int): Start index (inclusive)
        end_idx (int): End index (exclusive)
    
    Returns:
        str: Joined text
    """
    if start_idx >= end_idx:
        return ""
    return " ".join(words[start_idx:end_idx])


class GitSequenceMatcher:
    """
    Uses Git diff to compare sequences with move detection support.
    Returns opcodes in difflib format with an additional 'is_moved' flag.
    """
    def __init__(self, a, b, temp_dir=None):
        self.a = a
        self.b = b
        self.temp_file_a = None
        self.temp_file_b = None
        self.temp_dir = temp_dir

    def _create_temp_files(self):
        """Creates temporary files with repr() of each item in the input sequences."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', dir=self.temp_dir) as f_a:
            self.temp_file_a = f_a.name
            for item in self.a:
                f_a.write(repr(item) + '\n')

        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', dir=self.temp_dir) as f_b:
            self.temp_file_b = f_b.name
            for item in self.b:
                f_b.write(repr(item) + '\n')

    def _cleanup_temp_files(self):
        """Removes the temporary files."""
        if self.temp_file_a and os.path.exists(self.temp_file_a):
            os.remove(self.temp_file_a)
        if self.temp_file_b and os.path.exists(self.temp_file_b):
            os.remove(self.temp_file_b)

    def get_opcodes(self):
        """
        Generates a list of 6-tuple opcodes (tag, i1, i2, j1, j2, is_moved)
        similar to difflib.SequenceMatcher, with 'is_moved' flag for delete/insert.
        """
        self._create_temp_files()
        process = None
        start_time = time.time()
        try:
            command = [
                'git',
                '--no-pager',
                'diff',
                '--no-index',
                '--no-ext-diff',
                '--diff-algorithm=histogram',
                '--color=always',
                '--color-moved',
                '--unified=99999999',
                self.temp_file_a,
                self.temp_file_b
            ]
            print(f"\nRunning command: {' '.join(command)}")
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            diff_output = process.stdout

            if process.returncode == 0 and not diff_output.strip():
                with open(self.temp_file_a, 'r', encoding='utf-8', errors='replace') as f:
                    num_lines = sum(1 for _ in f)
                return [('equal', 0, num_lines, 0, num_lines, False)]

            COLOR_RED_FG = r'\x1b\[31m'
            COLOR_GREEN_FG = r'\x1b\[32m'
            COLOR_BOLD_MAGENTA_FG = r'\x1b\[1;35m'
            COLOR_BLUE_FG = r'\x1b\[1;34m'
            COLOR_BOLD_CYAN_FG = r'\x1b\[1;36m'
            COLOR_BOLD_YELLOW_FG = r'\x1b\[1;33m'
            COLOR_RED_BG = r'\x1b\[41m'

            current_a_idx = 0
            current_b_idx = 0
            lines = diff_output.splitlines()
            in_hunk = False
            granular_changes = []

            for line_num, line in enumerate(lines):
                line_without_ansi = re.sub(r'\x1b\[[0-9;]*m', '', line)

                if line.startswith('\x1b[1mdiff --git'):
                    in_hunk = True
                    continue
                if not in_hunk:
                    continue

                if line_without_ansi.strip().startswith('index ') or \
                   line_without_ansi.strip().startswith('--- a/') or \
                   line_without_ansi.strip().startswith('+++ b/'):
                    continue
                
                if line_without_ansi.strip().startswith('@@'):
                    match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line_without_ansi.strip())
                    if match:
                        current_a_idx = int(match.group(1)) - 1
                        current_b_idx = int(match.group(3)) - 1
                    continue
                
                tag = None
                content_to_match = ''

                if re.search(f'^{COLOR_BOLD_MAGENTA_FG}-', line) or re.search(f'^{COLOR_BLUE_FG}-', line):
                    tag = 'moved_delete'
                    content_to_match = line_without_ansi[1:].strip()
                elif re.search(f'^{COLOR_BOLD_CYAN_FG}\\+', line) or re.search(f'^{COLOR_BOLD_YELLOW_FG}\\+', line) or line.strip().endswith(f'{COLOR_RED_BG}'):
                    tag = 'moved_insert'
                    content_to_match = line_without_ansi[1:].strip()
                elif re.search(f'^{COLOR_RED_FG}-', line):
                    tag = 'delete'
                    content_to_match = line_without_ansi[1:].strip()
                elif re.search(f'^{COLOR_GREEN_FG}\\+', line):
                    tag = 'insert'
                    content_to_match = line_without_ansi[1:].strip()
                elif line_without_ansi.startswith(' '):
                    tag = 'equal'
                    content_to_match = line_without_ansi[1:].strip()
                else:
                    if not line_without_ansi.strip():
                        continue

                if tag and content_to_match:
                    if tag == 'delete' or tag == 'moved_delete':
                        granular_changes.append((tag, content_to_match, current_a_idx, current_a_idx + 1, current_b_idx, current_b_idx))
                        current_a_idx += 1
                    elif tag == 'insert' or tag == 'moved_insert':
                        granular_changes.append((tag, content_to_match, current_a_idx, current_a_idx, current_b_idx, current_b_idx + 1))
                        current_b_idx += 1
                    elif tag == 'equal':
                        granular_changes.append((tag, content_to_match, current_a_idx, current_a_idx + 1, current_b_idx, current_b_idx + 1))
                        current_a_idx += 1
                        current_b_idx += 1

            # Move Detection
            moved_candidates = {}
            for idx, (g_tag, g_content, g_a1, g_a2, g_b1, g_b2) in enumerate(granular_changes):
                if g_tag in ['moved_delete', 'moved_insert']:
                    if g_content not in moved_candidates:
                        moved_candidates[g_content] = []
                    moved_candidates[g_content].append((g_a1, g_b1, g_tag, idx))

            is_moved_flags = {}
            for content, candidates in moved_candidates.items():
                deletes = [c for c in candidates if c[2] == 'moved_delete']
                inserts = [c for c in candidates if c[2] == 'moved_insert']

                matched_deletes = set()
                matched_inserts = set()

                for d_a1, d_b1, d_tag, d_idx in deletes:
                    if d_idx in matched_deletes:
                        continue

                    for i_a1, i_b1, i_tag, i_idx in inserts:
                        if i_idx in matched_inserts:
                            continue
                        is_moved_flags[d_idx] = True
                        is_moved_flags[i_idx] = True
                        matched_deletes.add(d_idx)
                        matched_inserts.add(i_idx)
                        break

            # Consolidation
            final_opcodes_pre_replace = []
            current_tag = None
            current_i1, current_i2, current_j1, current_j2 = -1, -1, -1, -1
            current_is_moved_flag = False

            for idx, (g_tag, g_content, g_a1, g_a2, g_b1, g_b2) in enumerate(granular_changes):
                actual_tag = g_tag
                if actual_tag in ['moved_delete', 'moved_insert']:
                    actual_tag = 'delete' if g_tag == 'moved_delete' else 'insert'

                is_moved_for_this_item = is_moved_flags.get(idx, False)

                if current_tag is None:
                    current_tag = actual_tag
                    current_i1, current_i2 = g_a1, g_a2
                    current_j1, current_j2 = g_b1, g_b2
                    current_is_moved_flag = is_moved_for_this_item
                    continue

                can_extend = False
                if actual_tag == current_tag and is_moved_for_this_item == current_is_moved_flag:
                    if actual_tag == 'equal':
                        if g_a1 == current_i2 and g_b1 == current_j2:
                            can_extend = True
                    elif actual_tag == 'delete':
                        if g_a1 == current_i2:
                            can_extend = True
                    elif actual_tag == 'insert':
                        if g_b1 == current_j2:
                            can_extend = True
                
                if can_extend:
                    current_i2 = g_a2
                    current_j2 = g_b2
                else:
                    final_opcodes_pre_replace.append((current_tag, current_i1, current_i2, current_j1, current_j2, current_is_moved_flag))
                    current_tag = actual_tag
                    current_i1, current_i2 = g_a1, g_a2
                    current_j1, current_j2 = g_b1, g_b2
                    current_is_moved_flag = is_moved_for_this_item
            
            if current_tag is not None:
                final_opcodes_pre_replace.append((current_tag, current_i1, current_i2, current_j1, current_j2, current_is_moved_flag))

            # Merge adjacent delete/insert into replace
            consolidated_opcodes = []
            i = 0
            while i < len(final_opcodes_pre_replace):
                current_op = final_opcodes_pre_replace[i]
                tag, i1, i2, j1, j2, is_moved = current_op

                if (tag == 'delete' and not is_moved) and i + 1 < len(final_opcodes_pre_replace):
                    next_op = final_opcodes_pre_replace[i+1]
                    next_tag, next_i1, next_i2, next_j1, next_j2, next_is_moved = next_op

                    if (next_tag == 'insert' and not next_is_moved) and i2 == next_i1 and j2 == next_j1:
                        consolidated_opcodes.append(('replace', i1, i2, j1, next_j2, False))
                        i += 2
                        continue
                
                consolidated_opcodes.append(current_op)
                i += 1

            opcodes = sorted(consolidated_opcodes, key=lambda x: (x[1], x[3]))

        except Exception as e:
            print(f"An unexpected error occurred during parsing: {e}")
            traceback.print_exc()
            if process:
                print(f"Stderr from git: {process.stderr}")
            return []
        finally:
            self._cleanup_temp_files()
        
        print(f"Git diff completed in {time.time() - start_time:.2f} seconds")
        return opcodes


def is_git_diff_available():
    """Checks if git diff is available."""
    try:
        subprocess.run(['git', 'diff'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class PDFDiffExtractor:
    """
    A class to extract word-level differences between two PDF files.
    
    Usage:
        extractor = PDFDiffExtractor(pdf1_path, pdf2_path, output_json_path)
        result = extractor.run()
    
    Or with options:
        extractor = PDFDiffExtractor(
            pdf1_path, 
            pdf2_path, 
            output_json_path,
            case_insensitive=True,
            use_git_diff=True
        )
        result = extractor.run()
    """
    
    def __init__(
        self,
        pdf1_path: str,
        pdf2_path: str,
        output_path: str = None,
        case_insensitive: bool = False,
        ignore_quotes: bool = False,
        ignore_ligatures: bool = False,
        use_git_diff: bool = False,
        indent: int = 2
    ):
        """
        Initialize the PDF diff extractor.
        
        Args:
            pdf1_path: Path to the first PDF file
            pdf2_path: Path to the second PDF file
            output_path: Path to the output JSON file (optional, required for save() and run())
            case_insensitive: Ignore case differences in comparison
            ignore_quotes: Normalize different quote types
            ignore_ligatures: Don't expand ligatures when extracting text
            use_git_diff: Use git diff algorithm (supports move detection)
            indent: JSON indentation level (default: 2, use 0 for compact)
        """
        self.pdf1_path = Path(pdf1_path)
        self.pdf2_path = Path(pdf2_path)
        self.output_path = Path(output_path) if output_path else None
        self.case_insensitive = case_insensitive
        self.ignore_quotes = ignore_quotes
        self.ignore_ligatures = ignore_ligatures
        self.use_git_diff = use_git_diff
        self.indent = indent
        self.result = None
    
    def validate_inputs(self) -> bool:
        """Validate that input PDF files exist."""
        if not self.pdf1_path.exists():
            print(f"Error: File not found: {self.pdf1_path}", file=sys.stderr)
            return False
        if not self.pdf2_path.exists():
            print(f"Error: File not found: {self.pdf2_path}", file=sys.stderr)
            return False
        return True
    
    def compare(self) -> dict:
        """
        Compare the two PDFs and return the result dictionary.
        
        Returns:
            dict: Comparison results with opcodes and metadata, or None if failed
        """
        self.result = compare_pdfs(
            self.pdf1_path,
            self.pdf2_path,
            case_insensitive=self.case_insensitive,
            ignore_quotes=self.ignore_quotes,
            ignore_ligatures=self.ignore_ligatures,
            use_git_diff=self.use_git_diff
        )
        return self.result
    
    def save(self) -> bool:
        """
        Save the comparison result to the output JSON file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.output_path is None:
            print("Error: No output path specified.", file=sys.stderr)
            return False
        
        if self.result is None:
            print("Error: No comparison result to save. Run compare() first.", file=sys.stderr)
            return False
        
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                if self.indent == 0:
                    json.dump(self.result, f, ensure_ascii=False)
                else:
                    json.dump(self.result, f, ensure_ascii=False, indent=self.indent)
            print(f"\nResults saved to: {self.output_path}")
            return True
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            return False
    
    def get_summary(self) -> dict:
        """
        Get a summary of the comparison results.
        
        Returns:
            dict: Statistics about the comparison
        """
        if self.result is None:
            return {}
        
        stats = {"equal": 0, "replace": 0, "delete": 0, "insert": 0}
        for opcode in self.result["opcodes"]:
            tag = opcode["tag"]
            if tag in stats:
                stats[tag] += 1
        
        return stats
    
    def print_summary(self):
        """Print a summary of the comparison results."""
        stats = self.get_summary()
        if stats:
            print("\nSummary:")
            print(f"  Equal segments: {stats['equal']}")
            print(f"  Replacements: {stats['replace']}")
            print(f"  Deletions: {stats['delete']}")
            print(f"  Insertions: {stats['insert']}")
    
    def return_json(self) -> str:
        """
        Compare the two PDFs and return the result as a JSON string.
        
        This method validates inputs, runs the comparison, and returns the
        result as a JSON string without saving to a file.
        
        Returns:
            str: JSON string of comparison results, or None if failed
        """
        if not self.validate_inputs():
            return None
        
        result = self.compare()
        if result is None:
            print("Error: Comparison failed", file=sys.stderr)
            return None
        
        if self.indent == 0:
            return json.dumps(result, ensure_ascii=False)
        else:
            return json.dumps(result, ensure_ascii=False, indent=self.indent)
    
    def run(self) -> dict:
        """
        Run the full extraction pipeline: validate, compare, save, and print summary.
        
        Returns:
            dict: Comparison results, or None if failed
        """
        if not self.validate_inputs():
            return None
        
        result = self.compare()
        if result is None:
            print("Error: Comparison failed", file=sys.stderr)
            return None
        
        if not self.save():
            return None
        
        self.print_summary()
        return result


def compare_pdfs(pdf1_path, pdf2_path, case_insensitive=False, ignore_quotes=False, ignore_ligatures=False, use_git_diff=False):
    """
    Compare two PDFs and generate diff opcodes with text content.
    
    Args:
        pdf1_path (str): Path to first PDF
        pdf2_path (str): Path to second PDF
        case_insensitive (bool): Ignore case in comparison
        ignore_quotes (bool): Normalize quote types
        ignore_ligatures (bool): Don't expand ligatures
        use_git_diff (bool): Use git diff instead of difflib (supports move detection)
    
    Returns:
        dict: Comparison results with opcodes and metadata
    """
    print(f"Extracting words from '{pdf1_path}'...")
    words1 = extract_words_from_pdf(pdf1_path, ignore_ligatures)
    if words1 is None:
        return None
    
    print(f"Extracting words from '{pdf2_path}'...")
    words2 = extract_words_from_pdf(pdf2_path, ignore_ligatures)
    if words2 is None:
        return None
    
    print(f"Extracted {len(words1)} words from first PDF")
    print(f"Extracted {len(words2)} words from second PDF")
    
    # Normalize for comparison
    words1_compare = normalize_words(words1, case_insensitive, ignore_quotes)
    words2_compare = normalize_words(words2, case_insensitive, ignore_quotes)
    
    # Choose comparison method
    if use_git_diff:
        if not is_git_diff_available():
            print("Warning: git diff not available, falling back to difflib")
            use_git_diff = False
        else:
            print("Running git diff sequence matcher...")
            matcher = GitSequenceMatcher(words1_compare, words2_compare, temp_dir='.')
            opcodes_raw = matcher.get_opcodes()
    
    if not use_git_diff:
        print("Running difflib sequence matcher...")
        matcher = difflib.SequenceMatcher(None, words1_compare, words2_compare)
        opcodes_raw = matcher.get_opcodes()
    
    # Build result structure
    result = {
        "metadata": {
            "pdf1": str(pdf1_path),
            "pdf2": str(pdf2_path),
            "words_in_pdf1": len(words1),
            "words_in_pdf2": len(words2),
            "case_insensitive": case_insensitive,
            "ignore_quotes": ignore_quotes,
            "ignore_ligatures": ignore_ligatures,
            "comparison_method": "git_diff" if use_git_diff else "difflib"
        },
        "opcodes": []
    }
    
    # Convert opcodes to detailed format
    for opcode in opcodes_raw:
        if len(opcode) == 6:  # Git diff format with is_moved
            tag, i1, i2, j1, j2, is_moved = opcode
        else:  # Standard difflib format
            tag, i1, i2, j1, j2 = opcode
            is_moved = False
        
        opcode_entry = {
            "tag": tag,
            "i1": i1,
            "i2": i2,
            "j1": j1,
            "j2": j2,
            "text_pdf1": get_text_from_indices(words1, i1, i2),
            "text_pdf2": get_text_from_indices(words2, j1, j2)
        }
        
        if use_git_diff:
            opcode_entry["is_moved"] = is_moved
        
        result["opcodes"].append(opcode_entry)
    
    print(f"Generated {len(result['opcodes'])} opcodes")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract word-level differences between two PDFs and save to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s doc1.pdf doc2.pdf -o diff.json
  %(prog)s doc1.pdf doc2.pdf -o diff.json --case-insensitive
  %(prog)s doc1.pdf doc2.pdf -o diff.json --ignore-quotes --ignore-ligatures
  %(prog)s doc1.pdf doc2.pdf -o diff.json --use-git-diff
        """
    )
    
    parser.add_argument("pdf1", help="Path to first PDF file")
    parser.add_argument("pdf2", help="Path to second PDF file")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file path")
    parser.add_argument("--case-insensitive", action="store_true",
                        help="Ignore case differences in comparison")
    parser.add_argument("--ignore-quotes", action="store_true",
                        help="Normalize different quote types")
    parser.add_argument("--ignore-ligatures", action="store_true",
                        help="Don't expand ligatures when extracting text")
    parser.add_argument("--use-git-diff", action="store_true",
                        help="Use git diff algorithm (supports move detection)")
    parser.add_argument("--indent", type=int, default=2,
                        help="JSON indentation (default: 2, use 0 for compact)")
    
    args = parser.parse_args()
    
    # Validate input files
    pdf1_path = Path(args.pdf1)
    pdf2_path = Path(args.pdf2)
    
    if not pdf1_path.exists():
        print(f"Error: File not found: {pdf1_path}", file=sys.stderr)
        return 1
    
    if not pdf2_path.exists():
        print(f"Error: File not found: {pdf2_path}", file=sys.stderr)
        return 1
    
    # Compare PDFs
    result = compare_pdfs(
        pdf1_path,
        pdf2_path,
        case_insensitive=args.case_insensitive,
        ignore_quotes=args.ignore_quotes,
        ignore_ligatures=args.ignore_ligatures,
        use_git_diff=args.use_git_diff
    )
    
    if result is None:
        print("Error: Comparison failed", file=sys.stderr)
        return 1
    
    # Save to JSON
    output_path = Path(args.output)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            if args.indent == 0:
                json.dump(result, f, ensure_ascii=False)
            else:
                json.dump(result, f, ensure_ascii=False, indent=args.indent)
        print(f"\nResults saved to: {output_path}")
        
        # Print summary
        stats = {"equal": 0, "replace": 0, "delete": 0, "insert": 0}
        for opcode in result["opcodes"]:
            stats[opcode["tag"]] += 1
        
        print("\nSummary:")
        print(f"  Equal segments: {stats['equal']}")
        print(f"  Replacements: {stats['replace']}")
        print(f"  Deletions: {stats['delete']}")
        print(f"  Insertions: {stats['insert']}")
        
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
