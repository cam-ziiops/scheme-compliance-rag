"""PDF ingestion script for the RAG system."""

import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .config import (
    DOCS_DIR,
    CHROMADB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
)

console = Console()


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file, returning list of (page_num, text) tuples."""
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append((page_num, text))
        doc.close()
    except Exception as e:
        console.print(f"[red]Error reading {pdf_path.name}: {e}[/red]")
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def ingest_documents():
    """Ingest all PDFs from the docs directory into ChromaDB."""
    # Ensure directories exist
    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)

    # Get all PDF files
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {DOCS_DIR}[/yellow]")
        return

    console.print(f"[bold]Found {len(pdf_files)} PDF files to process[/bold]\n")

    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=str(CHROMADB_DIR))

    # Create embedding function using sentence-transformers
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Delete existing collection if it exists and create fresh
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # Collection doesn't exist yet

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "Scheme compliance documents"}
    )

    all_documents = []
    all_metadatas = []
    all_ids = []
    doc_id = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing PDFs...", total=len(pdf_files))

        for pdf_path in pdf_files:
            progress.update(task, description=f"Processing {pdf_path.name}...")

            # Extract text from each page
            pages = extract_text_from_pdf(pdf_path)

            for page_num, page_text in pages:
                # Chunk the page text
                chunks = chunk_text(page_text)

                for chunk_idx, chunk in enumerate(chunks):
                    all_documents.append(chunk)
                    all_metadatas.append({
                        "source": pdf_path.name,
                        "page": page_num,
                        "chunk_index": chunk_idx,
                    })
                    all_ids.append(f"doc_{doc_id}")
                    doc_id += 1

            progress.advance(task)

    # Add all documents to collection
    if all_documents:
        console.print(f"\n[bold]Embedding {len(all_documents)} chunks...[/bold]")

        # Add in batches to avoid memory issues
        batch_size = 100
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Adding to vector store...", total=len(all_documents))

            for i in range(0, len(all_documents), batch_size):
                batch_docs = all_documents[i:i + batch_size]
                batch_meta = all_metadatas[i:i + batch_size]
                batch_ids = all_ids[i:i + batch_size]

                collection.add(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids,
                )
                progress.advance(task, advance=len(batch_docs))

        console.print(f"\n[green]Successfully ingested {len(all_documents)} chunks from {len(pdf_files)} PDFs[/green]")
        console.print(f"[dim]Vector store saved to: {CHROMADB_DIR}[/dim]")
    else:
        console.print("[yellow]No text content found in PDFs[/yellow]")


if __name__ == "__main__":
    ingest_documents()
