#!/usr/bin/env python3
"""
Script to rank papers by relevance and extract the most relevant paragraphs from each paper.
"""
import os
import json
import csv
import argparse
import pdfplumber
import numpy as np
import torch
import logging
import nltk
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Try to download nltk data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PaperRanker")

class PaperRanker:
    def __init__(self, embedding_model='all-mpnet-base-v2', device=None):
        """
        Initialize the paper ranker.
        
        Args:
            embedding_model: Name of the SentenceTransformer model to use
            device: Device to run the model on ('cpu', 'cuda', 'mps'), None for auto-detection
        """
        # Set device (auto-detect if not specified)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {embedding_model} on {self.device}")
        self.model = SentenceTransformer(embedding_model, device=self.device)
        logger.info("Model loaded successfully")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Input text
            
        Returns:
            List of paragraphs
        """
        # Split by double newlines (common paragraph separator)
        paragraphs = [p.strip() for p in text.split("\n\n")]
        
        # Filter out empty paragraphs and those that are too short
        paragraphs = [p for p in paragraphs if p and len(p.split()) >= 10]
        
        # If we have very few paragraphs, try splitting by single newlines
        if len(paragraphs) < 5:
            paragraphs = [p.strip() for p in text.split("\n")]
            paragraphs = [p for p in paragraphs if p and len(p.split()) >= 10]
        
        # If still too few, use NLTK to split into sentences and group them
        if len(paragraphs) < 5 and text:
            sentences = nltk.sent_tokenize(text)
            paragraphs = []
            current_paragraph = []
            
            for sentence in sentences:
                current_paragraph.append(sentence)
                if len(current_paragraph) >= 3:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
            
            # Add any remaining sentences
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
        
        return paragraphs
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for a text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            if not text or text.isspace():
                return None
            
            # Truncate extremely long text
            if len(text) > 25000:
                text = text[:25000]
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Reshape embeddings for sklearn's cosine_similarity
        emb1 = np.array(embedding1).reshape(1, -1)
        emb2 = np.array(embedding2).reshape(1, -1)
        
        # Compute and return cosine similarity
        return float(cosine_similarity(emb1, emb2)[0][0])
    
    def analyze_paper(self, paper_info: Dict[str, str], query_embedding: np.ndarray, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Analyze a single paper.
        
        Args:
            paper_info: Dictionary with paper metadata
            query_embedding: Embedding of the query
            threshold: Similarity threshold
            
        Returns:
            Dictionary with analysis results
        """
        filepath = paper_info.get('FilePath', '')
        
        # Skip if not a PDF or file doesn't exist
        if not filepath or not filepath.lower().endswith('.pdf') or not os.path.exists(filepath):
            logger.warning(f"Skipping paper: {paper_info.get('Title', '')}, invalid or missing file")
            return None
        
        title = paper_info.get('Title', 'Unknown Title')
        logger.info(f"Analyzing paper: {title}")
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(filepath)
        if not text:
            logger.warning(f"No text extracted from {filepath}")
            return None
        
        # Split into paragraphs
        paragraphs = self.split_into_paragraphs(text)
        logger.info(f"Split into {len(paragraphs)} paragraphs")
        
        # Generate embeddings for each paragraph
        paragraph_data = []
        
        # Process paragraphs in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(paragraphs), batch_size):
            batch = paragraphs[i:i+batch_size]
            
            # Generate embeddings for the batch
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            
            # Process each paragraph in the batch
            for j, paragraph in enumerate(batch):
                embedding = batch_embeddings[j]
                
                # Compute similarity with query
                similarity = self.compute_similarity(embedding, query_embedding)
                
                # Add to results if above threshold
                if similarity >= threshold:
                    paragraph_data.append({
                        "content": paragraph,
                        "similarity": similarity
                    })
        
        # Sort paragraphs by similarity (highest first)
        paragraph_data.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Calculate overall paper relevance score (average of top 5 paragraphs or all if fewer)
        top_n = min(5, len(paragraph_data))
        if top_n > 0:
            overall_score = sum(p["similarity"] for p in paragraph_data[:top_n]) / top_n
        else:
            overall_score = 0.0
        
        # Keep only top paragraphs (maximum 10)
        paragraph_data = paragraph_data[:10]
        
        return {
            "title": title,
            "authors": paper_info.get('Author', 'Unknown'),
            "year": paper_info.get('Year', 'Unknown'),
            "doi": paper_info.get('DOI', ''),
            "filepath": filepath,
            "relevance_score": overall_score,
            "relevant_paragraphs": paragraph_data,
            "paragraph_count": len(paragraphs)
        }
    
    def rank_papers_from_csv(self, csv_path: str, query: str, output_path: str, threshold: float = 0.6) -> None:
        """
        Rank papers listed in a CSV file by relevance to a query.
        
        Args:
            csv_path: Path to the CSV file with paper information
            query: The query or focus for relevance comparison
            output_path: Path to the output JSON file
            threshold: Similarity threshold for paragraphs
        """
        logger.info(f"Ranking papers from CSV: {csv_path}")
        logger.info(f"Query: {query}")
        
        # Get embedding for the query
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            logger.error("Failed to get embedding for query")
            return
        
        # Read paper information from CSV
        papers = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    papers.append(row)
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            return
        
        logger.info(f"Found {len(papers)} papers in CSV")
        
        # Analyze each paper
        paper_results = []
        for paper in papers:
            # Skip papers that don't have 'Success' status
            if paper.get('Status', '') != 'Success':
                continue
                
            # Analyze paper
            result = self.analyze_paper(paper, query_embedding, threshold)
            if result is not None:
                paper_results.append(result)
        
        logger.info(f"Successfully analyzed {len(paper_results)} papers")
        
        # Sort papers by overall relevance score (highest first)
        paper_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Save results to JSON
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "query": query,
                    "threshold": threshold,
                    "total_papers_analyzed": len(paper_results),
                    "ranked_papers": paper_results
                }, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving JSON results: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Rank papers by relevance and extract relevant paragraphs")
    parser.add_argument('csv_file', help='Path to the CSV file with paper information')
    parser.add_argument('query', help='The query or focus for relevance comparison')
    parser.add_argument('--output', default='ranked_papers.json', help='Path to the output JSON file')
    parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold (default: 0.6)')
    parser.add_argument('--embedding-model', default='all-mpnet-base-v2', help='SentenceTransformer model to use')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None, help='Device to run the model on (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Initialize ranker
    ranker = PaperRanker(
        embedding_model=args.embedding_model,
        device=args.device
    )
    
    # Rank papers
    ranker.rank_papers_from_csv(
        csv_path=args.csv_file,
        query=args.query,
        output_path=args.output,
        threshold=args.threshold
    )
    
    print(f"Paper ranking complete. Results saved to {args.output}")
    
    # Print top 5 papers
    try:
        with open(args.output, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        if results and "ranked_papers" in results and results["ranked_papers"]:
            print("\nTop 5 Most Relevant Papers:")
            for i, paper in enumerate(results["ranked_papers"][:5], 1):
                print(f"{i}. {paper['title']} ({paper['year']})")
                print(f"   Relevance Score: {paper['relevance_score']:.4f}")
                print(f"   Authors: {paper['authors']}")
                if paper['doi']:
                    print(f"   DOI: {paper['doi']}")
                
                if paper['relevant_paragraphs']:
                    print(f"   Most Relevant Paragraph (Score: {paper['relevant_paragraphs'][0]['similarity']:.4f}):")
                    print(f"   \"{paper['relevant_paragraphs'][0]['content'][:200]}...\"")
                print()
    except Exception as e:
        print(f"Error displaying top papers: {str(e)}")
    
    return 0

if __name__ == "__main__":
    exit(main())