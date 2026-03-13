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

# Add the PDF-Diff-Functions directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PDF-Diff-Functions'))
from pdf_diff_extractor import PDFDiffExtractor


def extract_diffs(edits_list: list[dict]) -> list[dict]:
    extracted_diffs = []
    for edit in edits_list['opcodes']:
        if edit['tag'] == 'equal':
            continue
        

        # Check if the letters in text_pdf1 and text_pdf2 are the same after removing dashes and any whitespace in the middle
        if 'text_pdf1' in edit and 'text_pdf2' in edit:
            text1 = ''.join(edit['text_pdf1'].split()).replace('-', '')
            text2 = ''.join(edit['text_pdf2'].split()).replace('-', '')
            if text1 == text2:
                continue
        extracted_diffs.append(edit)
    return extracted_diffs


def extract_diffs_papers(papers_json: list[dict]) -> dict:
    """
    Extract diffs for all papers in the input list.
    
    Args:
        papers_json: List of paper dictionaries with arxiv_pdf_path and openreview_pdf_path
        
    Returns:
        A dictionary mapping paper IDs to their diffs
    """
    processed_count = 0
    skipped_count = 0
    error_count = 0

    diffs_dict = {}
    
    for datum in tqdm(papers_json, desc="Extracting diffs"):
        paper_id = datum.get('id', 'unknown')
        
        if datum.get('arxiv_pdf_path') is None or datum.get('openreview_pdf_path') is None:
            logger.debug(f"Skipping paper {paper_id}: missing PDF path(s)")
            skipped_count += 1
            diffs_dict[paper_id] = {
                'id': paper_id,
                'diffs': [],
            }
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
                diffs_dict[paper_id] = {
                    'id': paper_id,
                    'diffs': [],
                }
                continue
                
            edits_list = json.loads(result)
            edits_changes = extract_diffs(edits_list)  # Keep only opcodes with changes

            diffs_dict[paper_id] = {
                'id': paper_id,
                'diffs': edits_changes,
            }
            logger.debug(f"Extracted {len(edits_changes)} diffs for paper {paper_id}")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing paper {paper_id}: {e}")
            error_count += 1
            diffs_dict[paper_id] = {
                'id': paper_id,
                'diffs': [],
            }
    
    logger.info(f"Processing complete. Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")
    return diffs_dict


def get_dialogues(papers_json: list[dict]) -> dict:
    """
    Add OpenReview dialogues to each paper in the input list.
    
    Args:
        papers_json: List of paper dictionaries with 'id' field
    
    Returns:
        A dictionary mapping paper IDs to their dialogues
    """
    scraper = OpenReviewScraper()
    dialogue_dict = {}
    for datum in tqdm(papers_json, desc="Adding dialogues"):
        paper_id = datum.get('id', 'unknown')
        if datum.get('arxiv_pdf_path') is None or datum.get('openreview_pdf_path') is None:
            logger.debug(f"Skipping paper {paper_id}: missing PDF path(s)")
            dialogue_dict[paper_id] = {
                'id': paper_id,
                'dialogue': []
            }
            continue
        try:
            dialogue = scraper.fetch_reviewer_author_dialogues(paper_id)
            dialogue_dict[paper_id] = {
                'id': paper_id,
                'dialogue': dialogue
            }
            logger.debug(f"Added dialogue for paper {paper_id}")
        except Exception as e:
            logger.error(f"Error fetching dialogue for paper {paper_id}: {e}")
            dialogue_dict[paper_id] = {
                'id': paper_id,
                'dialogue': []
            }
    return dialogue_dict


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract diffs between two PDF files and save as JSON.")
    parser.add_argument("--input-file", type=str, help="Path to the input JSON file containing paper metadata.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--dont-diff", action='store_true', help="If set, skip the diff extraction step.")
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
        diffs_dict = extract_diffs_papers(papers_json)

        output_dirname = os.path.dirname(args.input_file) + '/diffs/'
        os.makedirs(output_dirname, exist_ok=True)
        
        output_file_name = os.path.join(output_dirname, os.path.basename(args.input_file).replace('.json', '_diffs.json'))
            
        logger.info(f"Saving output to: {output_file_name}")
            # Save output JSON
        with open(output_file_name, 'w') as f:
            json.dump(diffs_dict, f, indent=2)
        logger.info(f"Diffs extracted and saved to {output_file_name}")
    
    if args.dont_add_dialogues:
        pass
    else:
        input_file = args.input_file
        # Get the dialogue for each paper from openreview
        logger.info("Adding dialogues to papers")
        dialogue_dict = get_dialogues(json.load(open(input_file, 'r')))

        output_dirname = os.path.dirname(input_file) + '/dialogues/'
        os.makedirs(output_dirname, exist_ok=True)

        output_file_name = os.path.join(output_dirname, os.path.basename(input_file).replace('.json', '_dialogues.json'))
        logger.info(f"Saving output with dialogues to: {output_file_name}")
        with open(output_file_name, 'w') as f:
            json.dump(dialogue_dict, f, indent=2)
        logger.info(f"Dialogues added and saved to {output_file_name}")
