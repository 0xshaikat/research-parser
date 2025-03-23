import json
import os
import time
import argparse
import logging
import re
import random
import http.cookiejar
from urllib.request import Request, build_opener, HTTPCookieProcessor
from urllib.error import URLError, HTTPError
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_downloader.log"),
        logging.StreamHandler()
    ]
)

class AdvancedPaperDownloader:
    def __init__(self, json_file, output_dir="downloaded_papers", delay=2):
        self.json_file = json_file
        self.output_dir = output_dir
        self.delay = delay
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 15_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Mobile/15E148 Safari/604.1'
        ]
        
        # Enable cookie handling
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.cookie_jar))
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
    
    def load_json_data(self):
        """Load and parse the JSON file."""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {self.json_file}")
            return None
        except FileNotFoundError:
            logging.error(f"File not found: {self.json_file}")
            return None
    
    def generate_filename(self, paper):
        """Generate a filename for the paper based on its metadata."""
        if 'title' in paper:
            # Clean title to use as filename
            title = paper['title'].replace(':', '-').replace('/', '-')
            title = ''.join(c for c in title if c.isalnum() or c in ' -_')
            title = title.strip()
            
            # Add year if available
            if 'year' in paper:
                return f"{title}_{paper['year']}.pdf"
            return f"{title}.pdf"
        elif 'key' in paper:
            # Use the key as filename
            key = paper['key'].replace('/', '-').replace(':', '-')
            return f"{key}.pdf"
        else:
            # Generate a timestamp-based filename if no identifiers are available
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"paper_{timestamp}.pdf"
    
    def get_random_user_agent(self):
        """Get a random user agent from the list."""
        return random.choice(self.user_agents)
    
    def fetch_url(self, url, referer=None):
        """Fetch a URL with proper headers."""
        headers = {
            'User-Agent': self.get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        
        if referer:
            headers['Referer'] = referer
        
        request = Request(url, headers=headers)
        
        try:
            response = self.opener.open(request)
            return response.read(), response.geturl()
        except HTTPError as e:
            logging.error(f"HTTP Error ({e.code}): {url}")
            if e.code == 403:
                logging.info("Access forbidden. This might require institutional access or login.")
            return None, None
        except URLError as e:
            logging.error(f"URL Error: {e.reason} for {url}")
            return None, None
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return None, None
    
    def extract_pdf_link(self, html_content, base_url):
        """Extract PDF download link from HTML content."""
        if not html_content:
            return None
        
        # Convert bytes to string if needed
        if isinstance(html_content, bytes):
            html_content = html_content.decode('utf-8', errors='ignore')
        
        # Common patterns for PDF links
        pdf_patterns = [
            # ACM Digital Library pattern
            r'href=[\'"]([^\'"]*(?:fulltext|pdf|ft_gateway)[^\'"]*)[\'"]',
            # General PDF link patterns
            r'href=[\'"]([^\'"]*\.pdf[^\'"]*)[\'"]',
            r'href=[\'"]([^\'"]*(?:download|paper|article|fulltext|document)[^\'"]*)[\'"]',
            # DOI direct link pattern
            r'href=[\'"]([^\'"]*doi[^\'"]*)[\'"]',
        ]
        
        for pattern in pdf_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            if matches:
                # Get the first match and convert to absolute URL if needed
                pdf_link = matches[0]
                if not pdf_link.startswith(('http://', 'https://')):
                    pdf_link = urljoin(base_url, pdf_link)
                return pdf_link
        
        return None
    
    def download_paper(self, paper, output_path):
        """Try to download a paper using advanced techniques."""
        url = paper.get('url')
        doi = paper.get('doi')
        
        if not url and doi:
            url = f"https://doi.org/{doi}"
        
        if not url:
            logging.warning("No URL or DOI found for paper. Skipping.")
            return False
        
        # Step 1: First try to access the main article page
        logging.info(f"Attempting to access: {url}")
        html_content, final_url = self.fetch_url(url)
        
        if not html_content:
            logging.warning(f"Could not access {url}")
            return False
        
        # Step 2: Try to find PDF link on the page
        pdf_link = self.extract_pdf_link(html_content, final_url)
        
        if not pdf_link:
            logging.warning(f"Could not find PDF link on page: {final_url}")
            
            # Save the HTML content for debugging or manual inspection
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'wb') as f:
                f.write(html_content)
            logging.info(f"Saved HTML content to: {html_path}")
            
            return False
        
        # Step 3: Try to download the PDF
        logging.info(f"Found PDF link: {pdf_link}")
        time.sleep(1)  # Small delay before attempting download
        
        pdf_content, _ = self.fetch_url(pdf_link, referer=final_url)
        
        if not pdf_content:
            logging.warning(f"Could not download PDF from: {pdf_link}")
            return False
        
        # Check if content is likely a PDF (starts with %PDF)
        if pdf_content[:4] != b'%PDF':
            logging.warning("Downloaded content does not appear to be a PDF")
            
            # Save the content for inspection
            content_path = output_path.replace('.pdf', '.bin')
            with open(content_path, 'wb') as f:
                f.write(pdf_content)
            logging.info(f"Saved non-PDF content to: {content_path}")
            
            return False
        
        # Save the PDF
        with open(output_path, 'wb') as f:
            f.write(pdf_content)
        
        logging.info(f"Successfully downloaded PDF to: {output_path}")
        return True
    
    def download_papers(self):
        """Download all papers from the JSON file."""
        data = self.load_json_data()
        if not data:
            return
        
        # Check if we have search_results key or if the data is directly the list of papers
        if 'search_results' in data:
            papers = data['search_results']
        else:
            papers = data
        
        total_papers = len(papers)
        successful = 0
        failed = 0
        
        for i, paper in enumerate(papers):
            paper_title = paper.get('title', paper.get('key', 'Unknown'))
            logging.info(f"Processing paper {i+1}/{total_papers}: {paper_title}")
            
            filename = self.generate_filename(paper)
            output_path = os.path.join(self.output_dir, filename)
            
            # Skip if already downloaded
            if os.path.exists(output_path):
                logging.info(f"Paper already downloaded: {output_path}")
                successful += 1
                continue
            
            # Create metadata file regardless of download success
            metadata_path = output_path.replace('.pdf', '_metadata.txt')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"Title: {paper.get('title', 'Unknown')}\n")
                f.write(f"Authors: {paper.get('author', 'Unknown')}\n")
                f.write(f"Year: {paper.get('year', 'Unknown')}\n")
                f.write(f"URL: {paper.get('url', 'Unknown')}\n")
                f.write(f"DOI: {paper.get('doi', 'Unknown')}\n")
                
                if 'abstract' in paper and paper['abstract']:
                    f.write(f"\nAbstract:\n{paper['abstract']}\n")
            
            # Try to download
            success = self.download_paper(paper, output_path)
            
            if success:
                successful += 1
            else:
                failed += 1
                logging.warning(f"Failed to download paper: {paper_title}")
            
            # Add random delay between requests to avoid being blocked
            delay_time = self.delay + random.uniform(0.5, 2.0)
            logging.info(f"Waiting {delay_time:.2f} seconds before next request...")
            time.sleep(delay_time)
        
        logging.info(f"Download complete. Successfully downloaded {successful} out of {total_papers} papers.")
        logging.info(f"Failed to download {failed} papers.")
        
        return successful, failed, total_papers

def main():
    parser = argparse.ArgumentParser(description='Advanced research paper downloader.')
    parser.add_argument('json_file', help='Path to the JSON file containing paper information')
    parser.add_argument('--output-dir', default='downloaded_papers', help='Directory to save downloaded papers')
    parser.add_argument('--delay', type=float, default=2.0, help='Base delay between requests in seconds')
    
    args = parser.parse_args()
    
    downloader = AdvancedPaperDownloader(args.json_file, args.output_dir, args.delay)
    downloader.download_papers()

if __name__ == '__main__':
    main()