"""Configuration settings for the RAG system."""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Document and storage paths
DOCS_DIR = BASE_DIR / "docs"
CHROMADB_DIR = BASE_DIR / "data" / "chromadb"

# Chunking parameters
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters

# Embedding model (fast, good quality, runs locally)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB collection name
COLLECTION_NAME = "scheme_compliance"

# Query settings
DEFAULT_TOP_K = 5
