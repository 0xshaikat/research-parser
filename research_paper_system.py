#!/usr/bin/env python3
import os
import argparse
import logging
import json
import time
import datetime
from pathlib import Path

# Import your existing modules
# These will be imported from the files in the same directory
try:
    from advanced_paper_downloader import AdvancedPaperDownloader
    from paper_results_parser import parse_log_file, read_json_data, create_csv_report
except ImportError:
    print("Error: Could not import required modules.")
    print("Make sure advanced_paper_downloader.py and paper_results_parser.py are in the same directory.")
    exit(1)

class ResearchPaperSystem:
    def __init__(self, 
                 output_dir="research_papers", 
                 download_delay=3.0, 
                 log_file=None):
        """
        Initialize the research paper system.
        
        Args:
            output_dir (str): Directory to save downloaded papers and results
            download_delay (float): Delay between download requests
            log_file (str): Custom log file path (defaults to output_dir/logs/download_[timestamp].log)
        """
        self.output_dir = output_dir
        self.download_delay = download_delay
        
        # Create directory structure
        self.papers_dir = os.path.join(output_dir, "papers")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.reports_dir = os.path.join(output_dir, "reports")
        
        for directory in [self.papers_dir, self.logs_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set up logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_file or os.path.join(self.logs_dir, f"download_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("ResearchPaperSystem")
        self.logger.info(f"Initialized Research Paper System")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Log file: {self.log_file}")
    
    def download_papers_from_json(self, json_file):
        """
        Download papers from a JSON file.
        
        Args:
            json_file (str): Path to the JSON file containing paper information
            
        Returns:
            tuple: (successful, failed, total) counts of paper download attempts
        """
        self.logger.info(f"Starting paper download from JSON file: {json_file}")
        
        # Initialize the downloader
        downloader = AdvancedPaperDownloader(
            json_file=json_file,
            output_dir=self.papers_dir,
            delay=self.download_delay
        )
        
        # Start the download process
        successful, failed, total = downloader.download_papers()
        
        self.logger.info(f"Download complete. Successfully downloaded {successful} out of {total} papers.")
        self.logger.info(f"Failed to download {failed} papers.")
        
        return successful, failed, total
    
    def search_for_papers(self, query, limit=20):
        """
        Search for papers based on a query (placeholder for future implementation).
        
        Args:
            query (str): Research topic or search query
            limit (int): Maximum number of papers to retrieve
            
        Returns:
            str: Path to the generated JSON file
        """
        self.logger.info(f"Paper search requested for query: {query} (limit: {limit})")
        self.logger.warning("Paper search functionality is not yet implemented")
        
        # This is a placeholder for future implementation
        # In the future, this could use an LLM or web scraping to find papers
        
        # For now, create a dummy JSON file with the search query
        timestamp = int(time.time())
        json_file = os.path.join(self.output_dir, f"search_results_{timestamp}.json")
        
        # Create a minimal JSON structure
        search_data = {
            "search_time": timestamp,
            "search_query": query,
            "search_limit": limit,
            "search_results": []
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(search_data, f, indent=2)
        
        self.logger.info(f"Created placeholder JSON file: {json_file}")
        self.logger.info(f"To implement real search functionality, modify the search_for_papers method")
        
        return json_file
    
    def generate_report(self, json_file=None):
        """
        Generate a CSV report of download results.
        
        Args:
            json_file (str): Path to the JSON file (if None, will try to find the most recent one)
            
        Returns:
            str: Path to the generated CSV report
        """
        # If no JSON file specified, try to find the most recent one in the output directory
        if json_file is None:
            json_files = [f for f in os.listdir(self.output_dir) if f.endswith('.json')]
            if not json_files:
                self.logger.error("No JSON files found in the output directory")
                return None
            
            # Sort by modification time (most recent first)
            json_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.output_dir, f)), reverse=True)
            json_file = os.path.join(self.output_dir, json_files[0])
            self.logger.info(f"Using most recent JSON file: {json_file}")
        
        # Generate timestamp for the report filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(self.reports_dir, f"paper_report_{timestamp}.csv")
        
        try:
            self.logger.info(f"Parsing log file: {self.log_file}")
            log_results = parse_log_file(self.log_file)
            
            self.logger.info(f"Reading JSON data from: {json_file}")
            paper_dict, search_info, total_papers = read_json_data(json_file)
            
            self.logger.info(f"Creating CSV report: {csv_file}")
            create_csv_report(log_results, paper_dict, csv_file)
            
            # Print summary
            success_count = len(log_results['successful_downloads'])
            failed_count = len(log_results['failed_downloads'])
            processed_count = len(log_results['paper_titles'])
            
            self.logger.info("\nSummary:")
            self.logger.info(f"Total papers in JSON: {total_papers}")
            self.logger.info(f"Papers processed: {processed_count}")
            self.logger.info(f"Successfully downloaded: {success_count}")
            self.logger.info(f"Failed to download: {failed_count}")
            
            if search_info:
                self.logger.info("\nSearch Information:")
                self.logger.info(f"Search Time: {search_info['time']}")
                self.logger.info(f"Search URL: {search_info['url']}")
            
            self.logger.info(f"\nCSV report created successfully: {csv_file}")
            
            return csv_file
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return None
    
    def run_complete_workflow(self, input_source, is_query=False, limit=20):
        """
        Run the complete workflow: search/load papers, download, and generate report.
        
        Args:
            input_source (str): Either a JSON file path or a search query
            is_query (bool): Whether input_source is a search query (True) or JSON file (False)
            limit (int): Maximum number of papers to retrieve if is_query is True
            
        Returns:
            tuple: (json_file, csv_report) paths to the JSON file and CSV report
        """
        json_file = None
        
        # Step 1: Get paper metadata (either from JSON or search)
        if is_query:
            self.logger.info(f"Starting workflow with search query: {input_source}")
            json_file = self.search_for_papers(input_source, limit)
        else:
            self.logger.info(f"Starting workflow with JSON file: {input_source}")
            json_file = input_source
        
        if not json_file or not os.path.exists(json_file):
            self.logger.error(f"Invalid or missing JSON file: {json_file}")
            return None, None
        
        # Step 2: Download papers
        self.download_papers_from_json(json_file)
        
        # Step 3: Generate report
        csv_report = self.generate_report(json_file)
        
        self.logger.info("Complete workflow finished")
        return json_file, csv_report


def main():
    parser = argparse.ArgumentParser(description='Research Paper Download and Analysis System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--output-dir', default='research_papers', 
                              help='Directory to save downloaded papers and results')
    common_parser.add_argument('--delay', type=float, default=3.0, 
                              help='Delay between download requests in seconds')
    
    # Download from JSON command
    download_parser = subparsers.add_parser('download', parents=[common_parser],
                                          help='Download papers from a JSON file')
    download_parser.add_argument('json_file', help='Path to the JSON file containing paper information')
    
    # Search command
    search_parser = subparsers.add_parser('search', parents=[common_parser],
                                        help='Search for papers based on a query')
    search_parser.add_argument('query', help='Research topic or search query')
    search_parser.add_argument('--limit', type=int, default=20, 
                             help='Maximum number of papers to retrieve')
    
    # Report command
    report_parser = subparsers.add_parser('report', parents=[common_parser],
                                        help='Generate a report from download logs')
    report_parser.add_argument('--json-file', help='Path to the JSON file (if not specified, will use the most recent one)')
    report_parser.add_argument('--log-file', help='Path to the log file (if not specified, will use the default one)')
    
    # Complete workflow command
    workflow_parser = subparsers.add_parser('workflow', parents=[common_parser],
                                          help='Run the complete workflow')
    workflow_parser.add_argument('input', help='Either a JSON file path or a search query')
    workflow_parser.add_argument('--query', action='store_true', 
                               help='Treat input as a search query instead of a JSON file')
    workflow_parser.add_argument('--limit', type=int, default=20, 
                               help='Maximum number of papers to retrieve if using a search query')
    
    args = parser.parse_args()
    
    # Initialize the system
    system = ResearchPaperSystem(
        output_dir=args.output_dir,
        download_delay=args.delay,
        log_file=getattr(args, 'log_file', None)
    )
    
    # Execute the requested command
    if args.command == 'download':
        system.download_papers_from_json(args.json_file)
    
    elif args.command == 'search':
        json_file = system.search_for_papers(args.query, args.limit)
        print(f"Search results saved to: {json_file}")
    
    elif args.command == 'report':
        csv_report = system.generate_report(args.json_file)
        if csv_report:
            print(f"Report generated: {csv_report}")
    
    elif args.command == 'workflow':
        json_file, csv_report = system.run_complete_workflow(
            args.input, args.query, args.limit
        )
        if csv_report:
            print(f"Workflow completed successfully!")
            print(f"JSON file: {json_file}")
            print(f"CSV report: {csv_report}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()