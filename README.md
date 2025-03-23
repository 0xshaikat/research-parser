# Research Paper Analysis System

This system downloads academic papers from a JSON source and analyzes their content to identify relevant information for your research focus. It works in multiple stages: downloading papers, organizing paper metadata, and analyzing content for relevance.

## Setup

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)

### Installation

1. Clone this repository
   ```bash
   git clone <repository-url>
   cd research-parser
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## System Components

The system consists of three main components that work together to download, process, and analyze research papers:

### 1. Paper Downloader (`advanced_paper_downloader.py`)

This component handles the downloading of academic papers from online sources:

- **Input**: JSON file containing paper metadata (URLs, DOIs, titles, authors)
- **Processing**:
  - Parses the JSON to extract paper information
  - Simulates browser behavior to bypass publisher restrictions
  - Attempts to locate and download PDF files from publisher websites
  - Handles cookies, headers, and redirects to maximize download success
  - Implements random delays to avoid being blocked
- **Output**: 
  - Downloaded PDF files in the specified directory
  - Detailed logs of successful and failed downloads
  - Metadata files for papers where PDFs couldn't be accessed
- **Status**:
  - Of the 1000+ papers defined in Vladimir's JSON, only 55 were able to successfully be downloaded due to ACM restrictions

### 2. CSV Results Parser (`paper_results_parser.py`)

This component processes the download logs and creates a structured report:

- **Input**: Download log file and original JSON metadata
- **Processing**:
  - Extracts information about download attempts from the logs
  - Matches log entries with paper metadata from the JSON
  - Determines download status for each paper
  - Consolidates all information into a structured format
- **Output**:
  - CSV file containing comprehensive information about each paper
  - Status indicators (Success/Failed) for each download attempt
  - File paths to successfully downloaded PDFs
  - Complete metadata including titles, authors, DOIs, abstracts

### 3. Paper Ranking Analyzer (`paper_ranking_analyzer.py`)

This component analyzes the content of downloaded papers and ranks them by relevance:

- **Input**: CSV report from the previous component and a research query
- **Processing**:
  - Loads each PDF and extracts text content
  - Divides text into paragraphs of appropriate length
  - Uses a sentence transformer model to generate embeddings for paragraphs and the query
  - Computes semantic similarity between paragraphs and the research query
  - Ranks paragraphs by similarity score
  - Calculates an overall relevance score for each paper
- **Output**:
  - JSON file containing ranked papers by relevance score
  - Most relevant paragraphs from each paper with their similarity scores
  - Complete paper metadata for reference

## Usage

### Step 1: Download Papers

```bash
python advanced_paper_downloader.py results_1742406790.json --output-dir papers --delay 3
```

This downloads papers listed in the JSON file to the `papers` directory.

### Step 2: Generate CSV Report

```bash
python paper_results_parser.py advanced_downloader.log results_1742406790.json paper_results.csv
```

This creates a CSV report of all paper download attempts.

### Step 3: Analyze and Rank Papers

```bash
python paper_ranking_analyzer.py paper_results.csv "large language model multi-agent security risks alignment adversarial attacks safety vulnerabilities coordination autonomous agents collaborative AI" --device mps --output ranked_papers.json
```

This analyzes the papers, ranks them by relevance to your research focus, and extracts the most relevant paragraphs.

## Output Files

### From Paper Downloader
- **papers/**: Directory containing downloaded PDF files
- **advanced_downloader.log**: Log file with details of download attempts

### From CSV Results Parser
- **paper_results.csv**: CSV file with metadata for all papers and their download status

### From Paper Ranking Analyzer
- **ranked_papers.json**: JSON file containing:
  - Ranked list of papers by relevance
  - Overall relevance score for each paper
  - Most relevant paragraphs from each paper with similarity scores
  - Paper metadata (title, authors, year, DOI)

## Advanced Configuration

- **Adjust thresholds**: Lower the `--threshold` value (e.g., to 0.6) to include more results
- **Change embedding model**: Different models offer trade-offs between speed and accuracy
- **Device selection**: Use `--device cpu`, `--device cuda` (NVIDIA GPUs), or `--device mps` (Apple Silicon)

## Troubleshooting

- **Empty results**: If no papers meet the similarity threshold, try lowering the threshold value
- **Memory errors**: Process fewer papers at a time or reduce batch sizes
- **Embedding errors**: Check that the sentence-transformers package is properly installed
