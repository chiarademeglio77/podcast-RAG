import os
from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).parent.parent

# Data Directories
DATA_DIR = ROOT_DIR / "data"
ARTICLES_DIR = DATA_DIR / "articles"
VECTORDB_DIR = DATA_DIR / "vectordb"

# FAISS Database File
FAISS_INDEX_PATH = VECTORDB_DIR / "faiss_index"

# Settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# Ensure directories exist
ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
