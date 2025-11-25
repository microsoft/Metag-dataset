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
        
        if not self.username:
            raise ValueError('OPENREVIEW_USERNAME environment variable is not set')
        if not self.password:
            raise ValueError('OPENREVIEW_PASSWORD environment variable is not set')

    def fetch_papers(self) -> list:
        client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=self.username,
            password=self.password
        )

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape papers from OpenReview conference.')
    parser.add_argument('--conference_id', type=str, default='ICLR.cc/2024/Conference',
                        help='The conference ID to scrape papers from (default: ICLR.cc/2024/Conference)')
    args = parser.parse_args()

    if 'ICLR' in args.conference_id:
        venue = 'ICLR'

    scraper = OpenReviewScraper(conference_id=args.conference_id)
    scrape_file = scraper.run()

    # Run the Semantic Scholar indexer to find the semantic scholar paper
    # Use SS to find the arxiv link to the paper
    indexer = SemanticScholarIndexer(venue=venue)

    indexed_file = asyncio.run(indexer.run_all(scrape_file))
    