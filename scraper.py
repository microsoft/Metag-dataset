# Write a scraper that uses the OpenReview v2 API to scrape papers from a conference, default to ICLR 2024
import json
import argparse
import os
from datetime import datetime
import asyncio
import time

import requests
import openreview
from tqdm import tqdm
from semanticscholar import SemanticScholar
from typing import Optional
from semanticscholar.Paper import Paper
from Levenshtein import distance



class OpenReviewScraper:
    """
    Scrape OpenReview for papers from a given conference
    Write the details for ACCEPTED papers to a json file 
    """
    def __init__(self, conference_id='ICLR.cc/2024/Conference'):
        self.conference_id = conference_id
        self.base_url = 'https://api.openreview.net'
        self.output_dir = conference_id
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load credentials from environment variables
        self.username = os.environ.get('OPENREVIEW_USERNAME')
        self.password = os.environ.get('OPENREVIEW_PASSWORD')

        self.client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=self.username,
            password=self.password
        )
        
        if not self.username:
            raise ValueError('OPENREVIEW_USERNAME environment variable is not set')
        if not self.password:
            raise ValueError('OPENREVIEW_PASSWORD environment variable is not set')

    def fetch_papers(self) -> list:

        client = self.client

        reply_type = "Decision" # Check decisions for accepted papers

        submissions = client.get_all_notes(
                        invitation=f'{self.conference_id}/-/Submission',
                        details = 'replies')
        
        accepted_submissions = []

        for submission in tqdm(submissions, desc="Processing submissions"):
            replies = submission.details['replies']
            for reply in replies:
                if any(invitation.endswith(reply_type) for invitation in reply['invitations']):
                    decision = reply['content'].get('decision')['value']
                    if decision != 'Reject':
                        # Assert that 'Accept' is in the decision
                        assert 'Accept' in decision, f"Unexpected decision value: {decision}"
                        accepted_submissions.append(submission.to_json())     
                    break
        
        return accepted_submissions


    def save_papers(self, papers: list) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'papers_{timestamp}.json')
        with open(output_file, 'w') as f:
            json.dump(papers, f, indent=2)
        print(f'Saved {len(papers)} papers to {output_file}')
        return output_file


    def run(self) -> str:
        print(f'Scraping papers from {self.conference_id}...')
        papers = self.fetch_papers()
        output_file = self.save_papers(papers)
        print('Scraping completed.')
        return output_file


    def download_papers(self, input_file: str, output_dir: str = None):
        """
        Download PDFs for papers from the openreview scrape using authenticated client
        Args:
            input_file: Path to the JSON file containing paper metadata from OpenReview
            output_dir: Directory to save PDFs (defaults to same directory as input_file + '/PDFs')
        """
        # Use authenticated client for better rate limits
        client = self.client

        with open(input_file, 'r') as f:
            papers = json.load(f)
        
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_file), 'PDFs')
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = []

        for paper in tqdm(papers, desc="Downloading OpenReview PDFs", position=1, leave=True):
            paper_id = paper['id']
            
            # Add retry logic for rate limiting and connection errors
            max_retries = 5
            retry_delay = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    # Use client's get_pdf method - handles auth automatically
                    pdf_binary = client.get_pdf(paper_id)
                    
                    output_path = os.path.join(output_dir, f'{paper_id}_openreview.pdf')
                    with open(output_path, 'wb') as f:
                        f.write(pdf_binary)
                    
                    output_paths.append(output_path)
                    tqdm.write(f'Downloaded PDF for {paper_id} to {output_path}')
                    # Small delay to be respectful of API
                    time.sleep(0.5)
                    success = True
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    # Check for retryable errors: rate limit, connection issues, incomplete reads
                    is_retryable = (
                        '429' in error_msg or 
                        'rate limit' in error_msg or
                        'incompleteread' in error_msg or
                        'connection' in error_msg or
                        'timeout' in error_msg or
                        'broken' in error_msg
                    )
                    
                    if is_retryable and attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        tqdm.write(f'Error for {paper_id}: {e}. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})')
                        time.sleep(wait_time)
                    else:
                        tqdm.write(f'Failed to download PDF for {paper_id} after {attempt + 1} attempts: {e}')
                        break
            
            if not success:
                output_paths.append(None)
                    
        return output_dir, output_paths
    

    def fetch_reviews(self, id) -> list:
        """
        Fetch reviews for papers in the conference by id
        """
        client = self.client

        submission = client.get_note(id, details='replies')
        replies = submission.details['replies']
        for reply in replies:
            if any(invitation.endswith('Official_Review') for invitation in reply['invitations']):
                reviews = client.get_notes(invitation=reply['invitations'][0])
        
        return [review.to_json() for review in reviews]

    def fetch_reviewer_author_dialogues(self, note_id: str) -> list:
        """
        Extract the entire reviewer-author dialogue for a paper.
        Each dialogue thread (starting from a review) is a single entry.
        
        Args:
            note_id: The OpenReview note ID (forum ID) of the paper
            
        Returns:
            List of dialogue dictionaries, each containing:
            - reviewer_id: Anonymous reviewer identifier
            - review: The initial review content
            - dialogue: List of back-and-forth comments in chronological order
        """
        client = self.client
        
        # Get the submission with all replies
        submission = client.get_note(note_id, details='replies')
        replies = submission.details['replies']
        
        # Build a map of id -> reply for quick lookup
        reply_map = {reply['id']: reply for reply in replies}
        
        # Find all reviews (starting points for dialogues)
        reviews = []
        for reply in replies:
            if any('Official_Review' in inv for inv in reply['invitations']):
                reviews.append(reply)
        
        # Find all official comments
        comments = []
        for reply in replies:
            if any('Official_Comment' in inv for inv in reply['invitations']):
                comments.append(reply)
        
        # Build dialogue threads for each review
        dialogues = []
        
        for review in reviews:
            # Extract reviewer identifier from signatures
            reviewer_id = review['signatures'][0] if review['signatures'] else 'Unknown'
            # Extract just the reviewer number (e.g., "Reviewer_ABC1" from full path)
            if '/' in reviewer_id:
                reviewer_id = reviewer_id.split('/')[-1]
            
            # Build the dialogue thread
            dialogue_thread = []
            
            # Find all comments that are part of this review's thread
            # We need to build a tree and flatten it chronologically
            def get_thread_comments(parent_id: str) -> list:
                """Recursively get all comments in a thread"""
                thread = []
                for comment in comments:
                    if comment.get('replyto') == parent_id:
                        # Determine if this is an author or reviewer response
                        signatures = comment.get('signatures', [])
                        is_author = any('Authors' in sig for sig in signatures)
                        is_reviewer = any('Reviewer' in sig for sig in signatures)
                        
                        commenter_type = 'author' if is_author else ('reviewer' if is_reviewer else 'other')
                        commenter_id = signatures[0].split('/')[-1] if signatures else 'Unknown'
                        
                        comment_entry = {
                            'id': comment['id'],
                            'commenter_type': commenter_type,
                            'commenter_id': commenter_id,
                            'content': comment.get('content', {}),
                            'created_date': comment.get('cdate'),
                            'replyto': parent_id
                        }
                        thread.append(comment_entry)
                        
                        # Recursively get replies to this comment
                        thread.extend(get_thread_comments(comment['id']))
                
                return thread
            
            # Get all comments in this review's thread
            thread_comments = get_thread_comments(review['id'])
            
            # Sort by creation date
            thread_comments.sort(key=lambda x: x.get('created_date', 0) or 0)
            
            dialogue_entry = {
                'reviewer_id': reviewer_id,
                'review': {
                    'id': review['id'],
                    'content': review.get('content', {}),
                    'created_date': review.get('cdate')
                },
                'dialogue': thread_comments
            }
            
            dialogues.append(dialogue_entry)
        
        return dialogues

    def fetch_all_dialogues_for_papers(self, input_file: str, output_file: str = None) -> str:
        """
        Fetch reviewer-author dialogues for all papers in a JSON file.
        
        Args:
            input_file: Path to JSON file containing papers with 'id' field
            output_file: Path to save output (defaults to input_file with _dialogues suffix)
            
        Returns:
            Path to the output file
        """
        with open(input_file, 'r') as f:
            papers = json.load(f)
        
        results = []
        
        for paper in tqdm(papers, desc="Fetching dialogues"):
            paper_id = paper.get('id')
            title = paper.get('content', {}).get('title', {}).get('value', 'Unknown')
            
            try:
                dialogues = self.fetch_reviewer_author_dialogues(paper_id)
                results.append({
                    'paper_id': paper_id,
                    'title': title,
                    'dialogues': dialogues
                })
            except Exception as e:
                tqdm.write(f"Error fetching dialogues for {paper_id}: {e}")
                results.append({
                    'paper_id': paper_id,
                    'title': title,
                    'dialogues': [],
                    'error': str(e)
                })
            
            # Small delay to be respectful of API
            time.sleep(0.3)
        
        # Save results
        if output_file is None:
            output_file = input_file.replace('.json', '_with_dialogues.json')
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved dialogues for {len(results)} papers to {output_file}")
        return output_file


class SemanticScholarIndexer:
    """
    A class to look up papers on Semantic Scholar to identify correct arxiv submissions
    """
    def __init__(self, venue:str=None):
        self.client = SemanticScholar()
        self.venue = venue
        self.paper_match_threshold = 5  # Allow up to 5% difference in title matching based on Levenshtein distance
        self.last_request_time = 0  # Track last request time for rate limiting


    def searchpaper(self, title: str)-> Optional[dict]:
        """
        Search for a paper on Semantic Scholar by title
        """
        try:
            paper = self.client.search_paper(
                query=title,  
                match_title=True, # Match title
            )
            return paper.raw_data
        except Exception as e:
            print(f"Error searching for paper with title '{title}': {e}")
            return None


    async def searchpaper_api(self, title:str) -> Optional[dict]:
        """
        Search for a paper on Semantic Scholar by title using direct API call
        Rate limited to 1 request per second with retry logic for 429 errors
        """
        url = "https://api.semanticscholar.org/graph/v1/paper/search/match"  # Use /paper/search/match endpoint to search by title
        api_key = os.environ.get('SEMANTICSCHOLAR_API_KEY', None)
        if api_key is None:
            raise ValueError('SEMANTICSCHOLAR_API_KEY environment variable is not set')
        
        headers = {"x-api-key": api_key}
        params = {
            "query": title,
            "limit": 1,
            "fields": "paperId,title,externalIds,venue,publicationVenue",
            "venue": self.venue if self.venue else ""
        }
        
        max_retries = 5
        retry_delay = 2  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Rate limiting: ensure at least 1 second between requests
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                if time_since_last_request < 1.0:
                    await asyncio.sleep(1.0 - time_since_last_request)
                
                self.last_request_time = time.time()
                
                response = requests.get(url, params=params, headers=headers)
                
                if response.status_code == 429:
                    # Rate limit hit, wait and retry with exponential backoff
                    wait_time = retry_delay * (2 ** attempt)
                    tqdm.write(f"Rate limit hit (429) for '{title}'. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                if data['data']:
                    return data['data'][0]
                else:
                    return None
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    tqdm.write(f"Rate limit error for '{title}'. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    tqdm.write(f"HTTP error searching for paper with title '{title}': {e}")
                    return None
            except Exception as e:
                tqdm.write(f"Error searching for paper with title '{title}': {e}")
                return None
        
        tqdm.write(f"Max retries reached for '{title}'. Giving up.")
        return None    


    def assert_paper_match(self, ss_paper, openreview_paper:dict) -> bool:
        """
        Assert that the Semantic Scholar paper matches the OpenReview paper
        """
        if ss_paper is None:
            tqdm.write("Semantic Scholar paper is None")
            return False
        title = openreview_paper['content'].get('title').get('value')
        ss_title = ss_paper.get('title', '').lower()
        # Make sure titles match    

        # Check levenshtein distance or simple equality
        dist = distance(ss_title, title.lower())
        length = max(len(ss_title), len(title))
        dist = (dist / length) * 100  # Normalize distance as percentage
        if dist < self.paper_match_threshold:  # Allow up to threshold% difference
            return True
        else:
            return False

    
    async def run(self, openreview_paper: dict, ss_api_key_set: bool = False) -> Optional[dict]:
        title = openreview_paper['content'].get('title').get('value')
        
        try:
            if ss_api_key_set:
                ss_paper = await self.searchpaper_api(title)
                if ss_paper is None:
                    raise ValueError("No paper found via API")
            else:
                ss_paper = self.searchpaper(title)  # Search without API, use semantic scholar client
        except Exception as e:
            tqdm.write(f"Paper '{title}' does not exist on Semantic Scholar: {e}")
            return None 
        
        try:
            if self.assert_paper_match(ss_paper, openreview_paper):
                # tqdm.write(f"Paper matched: {title}")
                return ss_paper
        except AssertionError as e:
            tqdm.write(f"Paper mismatch for title: {title}. Error: {e}")
            return None


    def get_arxiv_id_from_ss(self, ss_paper: Paper) -> str:
        """
        Extract the arxiv link from the Semantic Scholar paper object
        """
        arxiv_id = ss_paper.get('externalIds', {}).get('ArXiv', None)
        return arxiv_id


    async def run_all(self, outfile: str) -> str:
        """
        For each OpenReview paper, find the corresponding Semantic Scholar paper
        and extract the arxiv link
        """
        with open(outfile) as f:
            openreview_papers = json.load(f)
        
        match_count = 0
        
        out_list = []

        if os.environ.get('SEMANTICSCHOLAR_API_KEY') is None:
            print("Running without API key may lead to rate limiting issues.")
            print("Consider setting an API key in the SEMANTICSCHOLAR_API_KEY environment variable.")
            ss_api_key_set = False
        else:
            ss_api_key_set = True

        for openreview_paper in tqdm(openreview_papers):
            ss_paper = await self.run(openreview_paper, ss_api_key_set=ss_api_key_set)
            if ss_paper:
                match_count += 1
                arxiv_id = self.get_arxiv_id_from_ss(ss_paper)
                openreview_paper['arxiv_id'] = arxiv_id
                out_list.append(openreview_paper)
            else:
                tqdm.write("Paper not found on Semantic Scholar, skipping...")

        print(f'Total matches found: {match_count}/{len(openreview_papers)}')

        # Save updated papers with arxiv links
        output_file = outfile.replace('.json', f'_with_arxiv.json')
        with open(output_file, 'w') as f:
            json.dump(out_list, f, indent=2)
        
        print(f'Saving {len(out_list)} updated papers with arxiv ids to {output_file}')    
        return output_file


class ArxivPDFDownloader:
    """
    Download PDFs from arxiv
    """

    def __init__(self):
        pass
    

    def get_arxiv_versions(self, arxiv_id: str) -> list:
        """
        Get all versions of an arxiv paper using the arxiv API
        Returns a list of version dictionaries with 'version' and 'created' fields
        """
        
        # Remove version suffix if present (e.g., '2301.12345v2' -> '2301.12345')
        base_arxiv_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
        
        try:
            return self._scrape_arxiv_versions(base_arxiv_id)
            
        except Exception as e:
            tqdm.write(f"Error fetching arxiv versions for {arxiv_id}: {e}")
            return []
    

    def _scrape_arxiv_versions(self, base_arxiv_id: str) -> list:
        """
        Scrape version history from arxiv website
        Returns list of dicts with 'version' (e.g., 'v1', 'v2') and 'submitted' (datetime string)
        """
        import re
        from bs4 import BeautifulSoup
        
        url = f'https://arxiv.org/abs/{base_arxiv_id}'
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the version history section
            version_section = soup.find('div', class_='submission-history')
            
            if not version_section:
                tqdm.write(f"No version history found for {base_arxiv_id}")
                return []
            
            versions = []
            # Get the full text content
            text = version_section.get_text()
            
            # Pattern: [v1] Wed, 24 May 2023 14:52:56 UTC
            # The text contains line breaks between versions
            pattern = r'\[v(\d+)\]\s+\w+,\s+(\d+\s+\w+\s+\d{4})\s+[\d:]+\s+UTC'
            matches = re.findall(pattern, text)
            
            for match in matches:
                version_num = match[0]
                date_str = match[1]
                
                # Parse date
                date_obj = datetime.strptime(date_str, '%d %b %Y')
                
                versions.append({
                    'version': f'v{version_num}',
                    'submitted': date_obj,
                    'yymm': date_obj.strftime('%y%m')
                })
            
            return versions
            
        except Exception as e:
            tqdm.write(f"Error scraping arxiv versions for {base_arxiv_id}: {e}")
            return []


    def get_version_before_submission(self, arxiv_id: str, submission_date_ms: int) -> Optional[str]:
        """
        For a paper with multiple arxiv versions, find the version submitted before 
        the conference submission date
        
        Args:
            arxiv_id: The arxiv ID (may include version like '2301.12345v2')
            submission_date_ms: Submission date in milliseconds since epoch (from OpenReview cdate)
        
        Returns:
            The arxiv ID with version suffix (e.g., '2301.12345v1') or None if no suitable version
        """
        base_arxiv_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
        
        # Convert submission date to datetime
        submission_date = datetime.fromtimestamp(submission_date_ms / 1000)
        
        # Get all versions
        versions = self.get_arxiv_versions(base_arxiv_id)
        
        if not versions:
            # If we can't get version info, skip this paper to avoid downloading
            # a version that may have been submitted after the conference deadline
            tqdm.write(f"Could not fetch version history for {base_arxiv_id}, skipping to avoid post-submission version")
            return None
        
        # Find the latest version submitted before the submission date
        suitable_versions = [
            v for v in versions 
            if v['submitted'] <= submission_date
        ]
        
        if not suitable_versions:
            # No version before submission date, return earliest version
            # earliest = min(versions, key=lambda x: x['submitted'])
            tqdm.write(f"No arxiv version before submission date {submission_date.strftime('%Y-%m-%d')} for {base_arxiv_id}.")
            return None
        
        # Get the latest version that's still before submission
        latest_suitable = max(suitable_versions, key=lambda x: x['submitted'])
        tqdm.write(f"Found arxiv {base_arxiv_id}{latest_suitable['version']} (submitted {latest_suitable['submitted'].strftime('%Y-%m-%d')}) before conference submission {submission_date.strftime('%Y-%m-%d')}")
        return f"{base_arxiv_id}{latest_suitable['version']}"


    def download_pdf(self, arxiv_id, output_dir, paper_id, submission_date_ms: Optional[int] = None):
        """
        Download the PDF for a given arxiv ID
        
        Args:
            arxiv_id: The arxiv ID
            output_dir: Directory to save the PDF
            paper_id: The paper ID for naming
            submission_date_ms: Optional submission date in milliseconds to download version before submission
        """
        # If submission date provided, find the appropriate version
        if submission_date_ms:
            arxiv_id = self.get_version_before_submission(arxiv_id, submission_date_ms)
            if not arxiv_id:
                # tqdm.write(f"Could not determine appropriate arxiv version for paper {paper_id}")
                return None
        
        pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
        response = requests.get(pdf_url)
        if response.status_code == 200:
            output_path = os.path.join(output_dir, f'{paper_id}_arxiv_{arxiv_id}.pdf')
            with open(output_path, 'wb') as f:
                f.write(response.content)
            tqdm.write(f'Downloaded arxiv PDF for {arxiv_id} to {output_path}')
            return output_path
        else:
            tqdm.write(f'Failed to download arxiv PDF for {arxiv_id}, status code {response.status_code}')
            return None


    def download_all_papers(self, json_file: str, output_dir: str = None, use_submission_date: bool = True):
        """
        Download PDFs for all papers in a JSON file with arxiv IDs
        
        Args:
            json_file: Path to JSON file containing papers with arxiv_id field
            output_dir: Directory to save PDFs (defaults to same directory as json_file + '/pdfs')
            use_submission_date: If True, downloads the arxiv version before submission date
        """
        # Load papers from JSON
        with open(json_file, 'r') as f:
            papers = json.load(f)
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(json_file), 'PDFs')
        
        os.makedirs(output_dir, exist_ok=True)
        
        successful_downloads = 0
        failed_downloads = 0
        output_paths = []  # Moved outside loop
        
        for paper in tqdm(papers, desc="Downloading arXiv PDFs", position=0, leave=True):
            arxiv_id = paper.get('arxiv_id')
            paper_id = paper.get('id')
            
            if not arxiv_id:
                tqdm.write(f"Skipping paper {paper_id}: No arxiv_id found")
                failed_downloads += 1
                output_paths.append(None)
                continue
            
            # Get submission date if using version control
            # cdate is the date from OpenReview in milliseconds that represents when the note was created
            submission_date_ms = paper.get('cdate') if use_submission_date else None

            try:
                result = self.download_pdf(arxiv_id, output_dir, paper_id, submission_date_ms)
                if result:
                    successful_downloads += 1
                    output_paths.append(result)
                else:
                    failed_downloads += 1
                    output_paths.append(None)
            except Exception as e:
                tqdm.write(f"Error downloading PDF for paper {paper_id} (arxiv: {arxiv_id}): {e}")
                failed_downloads += 1
                output_paths.append(None)
        
        print(f"\nDownload summary:")
        print(f"  Successful: {successful_downloads}")
        print(f"  Failed: {failed_downloads}")
        print(f"  Total: {len(papers)}")
        print(f"  PDFs saved to: {output_dir}")
        
        return output_dir, output_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape papers from OpenReview conference.')
    parser.add_argument('--conference_id', type=str, default='ICLR.cc/2024/Conference',
                        help='The conference ID to scrape papers from (default: ICLR.cc/2024/Conference)')
    
    parser.add_argument('--dont_scrape', action='store_true',
                        help="Don't scrape from OpenReview, use existing scrape file.")
    parser.add_argument('--scrape_file', type=str, default=None,
                        help='Path to an existing scrape file if not scraping anew.')
    
    parser.add_argument('--dont_index', action='store_true',
                        help="Don't perform indexing on Semantic Scholar to find arxiv links.")
    parser.add_argument('--index_file', type=str, default=None,
                        help='Path to an existing indexed file if not indexing anew.')
    
    parser.add_argument('--dont_download_pdfs', action='store_true',
                        help="Don't download arxiv PDFs for papers in the indexed file.")
    parser.add_argument('--pdf_input_file', type=str, default=None,
                        help='Path to JSON file with arxiv_id field to download PDFs from.')
    
    parser.add_argument('--pdf_output_dir', type=str, default=None,
                        help='Directory to save downloaded PDFs (defaults to input directory + /PDFs).')
    
    args = parser.parse_args()

    if 'ICLR' in args.conference_id:
        venue = 'ICLR'
    
    if '2024' in args.conference_id:
        year = '2024'

    if args.dont_scrape:
        scrape_file = args.scrape_file
    else: # Perform scraping 
        scraper = OpenReviewScraper(conference_id=args.conference_id)
        scrape_file = scraper.run()
        

    # Run the Semantic Scholar indexer to find the semantic scholar paper
    # Use SS to find the arxiv link to the paper
    if args.dont_index:
        indexed_file = args.index_file
    else:  # Perform indexing
        indexer = SemanticScholarIndexer(venue=venue)
        indexed_file = asyncio.run(indexer.run_all(scrape_file))
    
    # Download PDFs from arXiv if requested
    if args.dont_download_pdfs:
        pass
    else:
        print("Starting PDF download...")
        pdf_input = args.pdf_input_file if args.pdf_input_file else indexed_file
        if pdf_input is None:
            print("Error: No input file specified for PDF download. Use --pdf_input_file or run indexing first.")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            downloader = ArxivPDFDownloader()
            scraper = OpenReviewScraper(conference_id=args.conference_id)
            use_submission_date = True  # Always use submission date to get the version before submission
            
            # Run both downloads in parallel
            print("Starting parallel PDF downloads from arXiv and OpenReview...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both download tasks
                arxiv_future = executor.submit(
                    downloader.download_all_papers,
                    pdf_input,
                    output_dir=args.pdf_output_dir,
                    use_submission_date=use_submission_date
                )
                openreview_future = executor.submit(
                    scraper.download_papers,
                    pdf_input,
                    output_dir=args.pdf_output_dir
                )
                
                # Wait for both to complete and get results
                output_dir, output_paths_arxiv = arxiv_future.result()
                output_dir_openreview, output_paths_openreview = openreview_future.result()
            
            print("Parallel downloads completed.")

            # Update the input file JSON with the openreview and arxiv PDF paths
            json_data = []
            assert len(output_paths_arxiv) == len(output_paths_openreview), "Mismatch in number of downloaded PDFs"
            for arxiv_path, openreview_path, paper in zip(output_paths_arxiv, output_paths_openreview, json.load(open(pdf_input))):
                paper['arxiv_pdf_path'] = arxiv_path
                paper['openreview_pdf_path'] = openreview_path
                json_data.append(paper)
            
            # Save updated JSON
            updated_output_file = pdf_input.replace('.json', '_with_pdfs.json')

            with open(updated_output_file, 'w') as f:
                json.dump(json_data, f, indent=2)

    