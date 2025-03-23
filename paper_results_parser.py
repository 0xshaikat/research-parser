import argparse
import csv
import json
import os
import re
from datetime import datetime

def parse_log_file(log_file):
    """
    Parse the downloader log file to extract information about download attempts.
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        dict: Parsed information from the log file
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    results = {
        'successful_downloads': {},  # title -> filepath
        'failed_downloads': set(),   # titles of failed downloads
        'paper_urls': {},            # title -> url
        'paper_titles': [],          # ordered list of titles as they appear in the log
        'processing_order': {}       # title -> index
    }
    
    # Regular expression patterns
    success_pattern = re.compile(r'Successfully downloaded PDF to: (.*)')
    failed_pattern = re.compile(r'Failed to download paper: (.*)')
    processing_pattern = re.compile(r'Processing paper (\d+)/\d+: (.*)')
    url_pattern = re.compile(r'Attempting to access: (.*)')
    
    current_paper = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Check for paper being processed
            processing_match = processing_pattern.search(line)
            if processing_match:
                index = int(processing_match.group(1))
                title = processing_match.group(2)
                current_paper = title
                results['paper_titles'].append(title)
                results['processing_order'][title] = index
                continue
            
            # Check for URL being accessed
            url_match = url_pattern.search(line)
            if url_match and current_paper:
                results['paper_urls'][current_paper] = url_match.group(1)
                continue
            
            # Check for successful download
            success_match = success_pattern.search(line)
            if success_match:
                filepath = success_match.group(1)
                # Extract title from filepath
                title_match = re.search(r'papers/(.*?)\.pdf', filepath)
                if title_match and title_match.group(1):
                    paper_title = title_match.group(1).split('_')[0]  # Remove year suffix if present
                    # Find the most similar title
                    best_match = None
                    highest_similarity = 0
                    for known_title in results['paper_titles']:
                        if paper_title in known_title or known_title in paper_title:
                            similarity = len(paper_title) / max(len(known_title), 1)
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                best_match = known_title
                    
                    if best_match:
                        results['successful_downloads'][best_match] = filepath
                continue
            
            # Check for failed download
            failed_match = failed_pattern.search(line)
            if failed_match:
                results['failed_downloads'].add(failed_match.group(1))
                continue
    
    return results

def read_json_data(json_file):
    """
    Read and parse the JSON file containing paper metadata.
    
    Args:
        json_file (str): Path to the JSON file
        
    Returns:
        dict: Dictionary mapping paper titles to their metadata
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if we have search_results key or if the data is directly the list of papers
        if 'search_results' in data:
            papers = data['search_results']
            search_info = {
                'time': data.get('search_time', ''),
                'url': data.get('search_url', '')
            }
        else:
            papers = data
            search_info = None
        
        # Create a dictionary mapping titles to paper metadata
        paper_dict = {}
        for paper in papers:
            title = paper.get('title', '')
            if title:
                paper_dict[title] = paper
        
        return paper_dict, search_info, len(papers)
    
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {json_file}")

def create_csv_report(log_results, paper_dict, output_csv):
    """
    Create a CSV report combining log results and paper metadata.
    
    Args:
        log_results (dict): Results from parsing the log file
        paper_dict (dict): Paper metadata from the JSON file
        output_csv (str): Path to the output CSV file
    """
    fieldnames = [
        'Index', 'Title', 'Author', 'Year', 'URL', 'DOI', 'Journal', 
        'Type', 'Status', 'FilePath', 'Abstract'
    ]
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for title in log_results['paper_titles']:
            row = {
                'Index': log_results['processing_order'].get(title, ''),
                'Title': title,
                'URL': log_results['paper_urls'].get(title, ''),
                'FilePath': log_results['successful_downloads'].get(title, '')
            }
            
            # Set status
            if title in log_results['successful_downloads']:
                row['Status'] = 'Success'
            elif title in log_results['failed_downloads']:
                row['Status'] = 'Failed'
            else:
                row['Status'] = 'Unknown'
            
            # Add metadata from JSON
            if title in paper_dict:
                paper = paper_dict[title]
                row['Author'] = paper.get('author', '')
                row['Year'] = paper.get('year', '')
                row['DOI'] = paper.get('doi', '')
                row['Type'] = paper.get('type', '')
                row['Journal'] = paper.get('journal', '')
                
                # Truncate abstract if it's too long
                abstract = paper.get('abstract', '')
                if abstract and len(abstract) > 300:
                    row['Abstract'] = abstract[:297] + '...'
                else:
                    row['Abstract'] = abstract
            
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Parse paper download logs and generate a CSV report')
    parser.add_argument('log_file', help='Path to the download log file')
    parser.add_argument('json_file', help='Path to the JSON file containing paper metadata')
    parser.add_argument('output_csv', help='Path to the output CSV file')
    
    args = parser.parse_args()
    
    try:
        print(f"Parsing log file: {args.log_file}")
        log_results = parse_log_file(args.log_file)
        
        print(f"Reading JSON data from: {args.json_file}")
        paper_dict, search_info, total_papers = read_json_data(args.json_file)
        
        print(f"Creating CSV report: {args.output_csv}")
        create_csv_report(log_results, paper_dict, args.output_csv)
        
        # Print summary
        success_count = len(log_results['successful_downloads'])
        failed_count = len(log_results['failed_downloads'])
        processed_count = len(log_results['paper_titles'])
        
        print("\nSummary:")
        print(f"Total papers in JSON: {total_papers}")
        print(f"Papers processed: {processed_count}")
        print(f"Successfully downloaded: {success_count}")
        print(f"Failed to download: {failed_count}")
        
        if search_info:
            print("\nSearch Information:")
            print(f"Search Time: {search_info['time']}")
            print(f"Search URL: {search_info['url']}")
        
        print(f"\nCSV report created successfully: {args.output_csv}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())