# Write a scraper that uses the OpenReview v2 API to scrape papers from a conference, default to ICLR 2024
import json
import argparse
import os
from datetime import datetime
import openreview
from tqdm import tqdm
from semanticscholar import SemanticScholar
from typing import Optional
from semanticscholar.Paper import Paper


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


    def save_papers(self, papers) -> str:
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
    def __init__(self):
        self.client = SemanticScholar()


    def searchpaper(self, title: str)-> Optional[Paper]:
        """
        Search for a paper on Semantic Scholar by title
        """
        paper = self.client.search_paper(
            query=title,  
            match_title=True, # Match title
        )

        return paper


    def assert_paper_match(self, ss_paper, openreview_paper:dict) -> bool:
        """
        Assert that the Semantic Scholar paper matches the OpenReview paper
        """
        title = openreview_paper['content'].get('title').get('value')
        paper_bibtex = openreview_paper['content'].get('_bibtex').get('value')

        # Make sure titles match
        assert ss_paper.title.lower() == title.lower(), f"Title mismatch: {ss_paper.title} vs {title}"
        
        # Check if venues are the same 
        assert ss_paper.publicationVenue['name'] in paper_bibtex or any(alt_name in paper_bibtex for alt_name in ss_paper.publicationVenue['alternate_names']), f"Venue mismatch: {ss_paper.publicationVenue} not in {paper_bibtex}"
        return True

    
    def run(self, openreview_paper: dict):
        title = openreview_paper['content'].get('title').get('value')
        ss_paper = self.searchpaper(title)
        if ss_paper is None:
            print(f"No paper found on Semantic Scholar for title: {title}")
            return None
        try:
            if self.assert_paper_match(ss_paper, openreview_paper):
                print(f"Paper matched: {title}")
                return ss_paper
        except AssertionError as e:
            print(f"Paper mismatch for title: {title}. Error: {e}")
            return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape papers from OpenReview conference.')
    parser.add_argument('--conference_id', type=str, default='ICLR.cc/2024/Conference',
                        help='The conference ID to scrape papers from (default: ICLR.cc/2024/Conference)')
    args = parser.parse_args()

    scraper = OpenReviewScraper(conference_id=args.conference_id)
    outfile = scraper.run()

    indexer = SemanticScholarIndexer()
    with open(outfile) as f:
        paper_data = json.load(f)
    
    for paper_datum in tqdm(paper_data):
        try:
            indexer.run(paper_datum)
        except Exception as e:
            print(f"Error indexing paper: {e}")
            import bpdb; bpdb.set_trace()
    
    print('Indexing completed successfully.')