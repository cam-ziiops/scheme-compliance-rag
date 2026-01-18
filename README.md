# Scheme Compliance RAG

A local RAG (Retrieval-Augmented Generation) system for querying Visa/Mastercard rules and compliance documents.

## Features

- **Local embeddings** using `all-MiniLM-L6-v2` (no API costs)
- **Persistent vector storage** with ChromaDB
- **PDF text extraction** with PyMuPDF
- **Source citations** with file name, page number, and similarity score
- **Interactive CLI** with rich formatting

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Add PDF Documents

Place your PDF files in the `docs/` directory.

### 2. Ingest Documents

Run the ingestion script to extract text, create embeddings, and store them in ChromaDB:

```bash
python -m rag.ingest
```

This only needs to be run once, or when documents change.

### 3. Query the Knowledge Base

**Single query:**
```bash
python -m rag.query "What is PCI DSS?"
```

**Interactive mode:**
```bash
python -m rag.query --interactive
```

**Custom number of results:**
```bash
python -m rag.query -k 10 "chargeback timeframes"
```

## Project Structure

```
Scheme Compliance/
├── docs/                    # PDF documents to index
├── rag/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── ingest.py           # PDF ingestion script
│   └── query.py            # CLI query interface
├── data/
│   └── chromadb/           # Persistent vector store
├── requirements.txt
└── README.md
```

## Configuration

Edit `rag/config.py` to customize:

- `CHUNK_SIZE` - Characters per chunk (default: 1000)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 200)
- `EMBEDDING_MODEL` - Sentence transformer model (default: `all-MiniLM-L6-v2`)
- `DEFAULT_TOP_K` - Number of results to return (default: 5)

## License

MIT
