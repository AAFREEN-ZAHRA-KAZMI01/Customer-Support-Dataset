import pandas as pd
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load the FAQ dataset from CSV and convert to list of dictionaries."""
    df = pd.read_csv(file_path)
    return df.to_dict('records')

def chunk_data(
    data: List[Dict[str, Any]], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Split FAQ entries into smaller chunks for better retrieval.
    
    Args:
        data: List of FAQ items (each with Question, Answer, Category)
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
    
    Returns:
        List of chunked documents with metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = []
    for item in data:
        # Combine question and answer for chunking
        text = f"Question: {item['Question']}\nAnswer: {item['Answer']}"
        splits = text_splitter.split_text(text)
        
        for i, split in enumerate(splits):
            chunk = {
                "text": split,
                "metadata": {
                    "category": item["Category"],
                    "original_question": item["Question"],
                    "chunk_num": i + 1,
                    "total_chunks": len(splits)
                }
            }
            chunks.append(chunk)
    
    return chunks