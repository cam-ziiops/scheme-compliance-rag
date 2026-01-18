"""CLI query interface for the RAG system."""

import argparse
import sys

import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from .config import (
    CHROMADB_DIR,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
)

console = Console()


def get_collection():
    """Get the ChromaDB collection."""
    if not CHROMADB_DIR.exists():
        console.print("[red]Error: Vector store not found. Run 'python -m rag.ingest' first.[/red]")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(CHROMADB_DIR))

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except ValueError:
        console.print("[red]Error: Collection not found. Run 'python -m rag.ingest' first.[/red]")
        sys.exit(1)

    return collection


def query(question, top_k=DEFAULT_TOP_K):
    """Query the knowledge base and return relevant chunks."""
    collection = get_collection()

    results = collection.query(
        query_texts=[question],
        n_results=top_k,
    )

    return results


def display_results(question, results):
    """Display query results with nice formatting."""
    console.print(Panel(question, title="Query", border_style="blue"))
    console.print()

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        console.print("[yellow]No relevant results found.[/yellow]")
        return

    console.print(f"[bold]Found {len(documents)} relevant chunks:[/bold]\n")

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        similarity = 1 - dist  # Convert distance to similarity score

        # Truncate long chunks for display
        display_text = doc[:500] + "..." if len(doc) > 500 else doc

        console.print(Panel(
            display_text,
            title=f"[bold cyan]Result {i}[/bold cyan] | {source} (Page {page})",
            subtitle=f"Similarity: {similarity:.2%}",
            border_style="dim",
        ))
        console.print()


def interactive_mode():
    """Run interactive query mode."""
    console.print(Panel.fit(
        "[bold]Scheme Compliance Knowledge Base[/bold]\n"
        "Enter your questions below. Type 'quit' or 'exit' to stop.",
        border_style="green"
    ))
    console.print()

    collection = get_collection()
    doc_count = collection.count()
    console.print(f"[dim]Loaded {doc_count} document chunks[/dim]\n")

    while True:
        try:
            question = console.input("[bold green]Question:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        results = collection.query(
            query_texts=[question],
            n_results=DEFAULT_TOP_K,
        )

        console.print()
        display_results(question, results)


def main():
    parser = argparse.ArgumentParser(
        description="Query the Scheme Compliance knowledge base"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="The question to ask (omit for interactive mode)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Enter interactive mode for multiple queries",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K})",
    )

    args = parser.parse_args()

    if args.interactive or not args.question:
        interactive_mode()
    else:
        results = query(args.question, args.top_k)
        display_results(args.question, results)


if __name__ == "__main__":
    main()
