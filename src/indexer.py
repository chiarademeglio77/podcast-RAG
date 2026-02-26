import os
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Tuple
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import src.config as config

class Indexer:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        self.filename_regex = re.compile(r"^(\d{2}-\d{2}-\d{2})_(.*?)_(.*)\.txt$")

    def parse_filename(self, filename: str) -> Dict:
        """Extract date, source, and title from filename."""
        match = self.filename_regex.match(filename)
        if match:
            date_str, source, title = match.groups()
            try:
                # Convert YY-MM-DD to datetime object
                date_dt = datetime.strptime(date_str, "%y-%m-%d")
                # Store as timestamp (string) for FAISS/JSON compatibility
                date_iso = date_dt.date().isoformat()
            except ValueError:
                date_iso = "unknown"
            
            return {
                "date": date_iso,
                "source": source.strip(),
                "title": title.strip()
            }
        return {
            "date": "unknown",
            "source": "unknown",
            "title": filename
        }

    def index_files(self, directory: str = str(config.ARTICLES_DIR)):
        """Loader and indexer for text files with metadata parsing."""
        print(f"Scanning directory: {directory}")
        
        txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
        if not txt_files:
            print("No text files found to index.")
            return

        all_documents = []
        print(f"Parsing metadata for {len(txt_files)} files...")
        
        for filename in tqdm(txt_files):
            filepath = os.path.join(directory, filename)
            metadata = self.parse_filename(filename)
            metadata["filename"] = filename
            metadata["source_path"] = filepath
            
            # Load content
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Create LangChain documents manually to inject metadata
                doc_chunks = self.text_splitter.create_documents(
                    [content], 
                    metadatas=[metadata]
                )
                all_documents.extend(doc_chunks)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        if not all_documents:
            print("No documents were successfully processed.")
            return

        print(f"Creating FAISS index with {len(all_documents)} chunks...")
        vector_db = FAISS.from_documents(all_documents, self.embeddings)
        
        # Save the index locally
        vector_db.save_local(str(config.FAISS_INDEX_PATH))
        print(f"Indexing complete. Index saved to {config.FAISS_INDEX_PATH}")

if __name__ == "__main__":
    indexer = Indexer()
    indexer.index_files()
