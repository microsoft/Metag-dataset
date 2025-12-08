# Extract diffs from the PDFs
from argparse import ArgumentParser
import os
import sys
import json
from tqdm import tqdm

# Add the PDF-Diff-Viewer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PDF-Diff-Viewer'))
from pdf_diff_extractor import PDFDiffExtractor


def extract_diffs(edits_list: list[dict]) -> list[dict]:
    extracted_diffs = []
    for edit in edits_list['opcodes']:
        if edit['tag'] == 'equal':
            continue
        extracted_diffs.append(edit)
    return extracted_diffs





if __name__ == "__main__":
    parser = ArgumentParser(description="Extract diffs between two PDF files and save as JSON.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSON file containing paper metadata.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output JSON file to save diffs.")
    args = parser.parse_args()

    # Load input JSON
    with open(args.input_file, 'r') as f:
        papers_json = json.load(f)
    
    for datum in papers_json:
        if datum['arxiv_pdf_path'] is None or datum['openreview_pdf_path'] is None:
            continue    
        extractor = PDFDiffExtractor(
            pdf1_path=datum['arxiv_pdf_path'],
            pdf2_path=datum['openreview_pdf_path'],
            # output_path=None,
            use_git_diff=True,
        )
        result = extractor.return_json()
        edits_list = json.loads(result)
        import bpdb; bpdb.set_trace() 