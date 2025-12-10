# Extract diffs from the PDFs and add reviewer-author dialogues from OpenReview
from argparse import ArgumentParser
import os
import sys
import json
import logging
from tqdm import tqdm
from scraper import OpenReviewScraper 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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


def process_papers(papers_json: list[dict]) -> list[dict]:
    """
    Extract diffs for all papers in the input list.
    
    Args:
        papers_json: List of paper dictionaries with arxiv_pdf_path and openreview_pdf_path
        
    Returns:
        The input list with 'diffs' field added to each paper
    """
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for datum in tqdm(papers_json, desc="Extracting diffs"):
        paper_id = datum.get('id', 'unknown')
        
        if datum.get('arxiv_pdf_path') is None or datum.get('openreview_pdf_path') is None:
            logger.debug(f"Skipping paper {paper_id}: missing PDF path(s)")
            skipped_count += 1
            continue
        
        try:
            logger.debug(f"Processing paper {paper_id}")
            logger.debug(f"  arxiv: {datum['arxiv_pdf_path']}")
            logger.debug(f"  openreview: {datum['openreview_pdf_path']}")
            
            extractor = PDFDiffExtractor(
                pdf1_path=datum['arxiv_pdf_path'],
                pdf2_path=datum['openreview_pdf_path'],
                use_git_diff=True,
            )
            result = extractor.return_json()
            
            if result is None:
                logger.warning(f"Failed to extract diffs for paper {paper_id}: extractor returned None")
                error_count += 1
                datum['diffs'] = []
                continue
                
            edits_list = json.loads(result)
            edits_changes = extract_diffs(edits_list)  # Keep only opcodes with changes

            datum['diffs'] = edits_changes
            logger.debug(f"Extracted {len(edits_changes)} diffs for paper {paper_id}")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing paper {paper_id}: {e}")
            error_count += 1
            datum['diffs'] = []
    
    logger.info(f"Processing complete. Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")
    return papers_json


def add_dialogues(papers_json: list[dict]) -> list[dict]:
    """
    Add OpenReview dialogues to each paper in the input list.
    
    Args:
        papers_json: List of paper dictionaries with 'id' field
    
    Returns:
        The input list with 'dialogue' field added to each paper
    """
    scraper = OpenReviewScraper()
    for datum in tqdm(papers_json, desc="Adding dialogues"):
        paper_id = datum.get('id', 'unknown')
        try:
            dialogue = scraper.fetch_reviewer_author_dialogues(paper_id)
            datum['dialogue'] = dialogue
            logger.debug(f"Added dialogue for paper {paper_id}")
        except Exception as e:
            logger.error(f"Error fetching dialogue for paper {paper_id}: {e}")
            datum['dialogue'] = []
    return papers_json


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract diffs between two PDF files and save as JSON.")
    parser.add_argument("--input-file", type=str, help="Path to the input JSON file containing paper metadata.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--dont-diff", action='store_true', help="If set, skip the diff extraction step.")
    parser.add_argument("--file-with-diff", type=str, help="If set, use this file as the output file from the diff extraction step.")
    parser.add_argument("--dont-add-dialogues", action='store_true', help="If set, skip the dialogue addition step.")
    args = parser.parse_args()

    logger.info(f"Loading input file: {args.input_file}")
    
    if args.dont_diff:
        pass
    else:
        # Load input JSON
        if args.input_file is None:
            logger.error("Input file is required unless --dont-diff is set.")
            logger.error("Run scraper.py first to generate the input JSON file.")
            sys.exit(1)
        with open(args.input_file, 'r') as f:
            papers_json = json.load(f)

        logger.info(f"Loaded {len(papers_json)} papers from input file")

        # Extract the diffs for each paper
        papers_json = process_papers(papers_json)
            
        output_file_name = args.input_file.replace('.json', '_with_diffs.json')
        logger.info(f"Saving output to: {output_file_name}")
            # Save output JSON
        with open(output_file_name, 'w') as f:
            json.dump(papers_json, f, indent=2)
        logger.info(f"Diffs extracted and saved to {output_file_name}")
    
    if args.dont_add_dialogues:
        pass
    else:
        input_file = args.file_with_diff if args.file_with_diff else args.input_file.replace('.json', '_with_diffs.json')

        # Get the dialogue for each paper from openreview
        logger.info("Adding dialogues to papers")
        papers_json = add_dialogues(json.load(open(input_file, 'r')))
        output_file_name = input_file.replace('.json', '_with_dialogues.json')
        logger.info(f"Saving output with dialogues to: {output_file_name}")
        with open(output_file_name, 'w') as f:
            json.dump(papers_json, f, indent=2)
        logger.info(f"Dialogues added and saved to {output_file_name}")
