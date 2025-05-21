import os
import glob
from typing import List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_chatbot.agents.query_agent import query_agent, RAGQueryAgentInputSchema, RAGQueryAgentOutputSchema
from rag_chatbot.agents.qa_agent import qa_agent, RAGQuestionAnsweringAgentInputSchema, RAGQuestionAnsweringAgentOutputSchema
from rag_chatbot.context_providers import RAGContextProvider, ChunkItem
from rag_chatbot.services.chroma_db import ChromaDBService
from rag_chatbot.config import CHUNK_SIZE, CHUNK_OVERLAP, NUM_CHUNKS_TO_RETRIEVE, CHROMA_PERSIST_DIR


console = Console()

WELCOME_MESSAGE = """
Welcome to the Markdown RAG Chatbot! I can help you find information from your markdown documentation.
Ask me any questions about the content in your markdown files and I'll use my knowledge base to provide accurate answers.

I'll show you my thought process:
1. First, I'll generate a semantic search query from your question
2. Then, I'll retrieve relevant chunks of text from the documentation
3. Finally, I'll analyze these chunks to provide you with an answer
"""

STARTER_QUESTIONS = [
    "What is the main purpose of this project?",
    "How do I get started with this project?",
    "What are the key components of the system?",
]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split the text into chunks with overlap."""
    # Split into paragraphs first
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    current_size = 0

    for i, paragraph in enumerate(paragraphs):
        if current_size + len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Include some overlap from the previous chunk
            if overlap > 0 and chunks:
                last_chunk = chunks[-1]
                overlap_text = " ".join(last_chunk.split()[-overlap:])
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                current_chunk = paragraph
            current_size = len(current_chunk)
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            current_size += len(paragraph)

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def load_markdown_files(docs_dir: str) -> List[Tuple[str, str]]:
    """Load all markdown files from the specified directory and its subdirectories.
    
    Args:
        docs_dir: Path to the directory containing markdown files
        
    Returns:
        List of tuples containing (file_path, file_content)
    """
    console.print(f"\n[bold yellow]📚 Loading markdown files from {docs_dir}...[/bold yellow]")
    
    # Ensure the directory exists
    if not os.path.exists(docs_dir):
        console.print(f"[bold red]Error: Directory {docs_dir} does not exist![/bold red]")
        return []
    
    # Find all markdown files recursively
    markdown_files = glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True)
    
    if not markdown_files:
        console.print(f"[bold yellow]Warning: No markdown files found in {docs_dir}[/bold yellow]")
        return []
    
    console.print(f"[dim]• Found {len(markdown_files)} markdown files[/dim]")
    
    # Load content from each file
    file_contents = []
    for file_path in markdown_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                file_contents.append((file_path, content))
                console.print(f"[dim]• Loaded {os.path.relpath(file_path, docs_dir)}[/dim]")
        except Exception as e:
            console.print(f"[bold red]Error loading {file_path}: {str(e)}[/bold red]")
    
    console.print(f"[bold green]✓ Successfully loaded {len(file_contents)} markdown files![/bold green]")
    return file_contents


def process_markdown_files(markdown_files: List[Tuple[str, str]], chunk_size: int, overlap: int) -> Tuple[List[str], List[dict]]:
    """Process markdown files into chunks with metadata.
    
    Args:
        markdown_files: List of tuples containing (file_path, file_content)
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        Tuple of (chunks, metadatas)
    """
    console.print("\n[bold yellow]🔄 Processing markdown files...[/bold yellow]")
    
    all_chunks = []
    all_metadatas = []
    
    for file_path, content in markdown_files:
        # Extract file name for metadata
        file_name = os.path.basename(file_path)
        
        # Chunk the content
        file_chunks = chunk_text(content, chunk_size, overlap)
        
        # Create metadata for each chunk
        file_metadatas = [
            {
                "source": file_path,
                "file_name": file_name,
                "chunk_index": i
            } 
            for i in range(len(file_chunks))
        ]
        
        all_chunks.extend(file_chunks)
        all_metadatas.extend(file_metadatas)
        
        console.print(f"[dim]• Processed {file_name}: created {len(file_chunks)} chunks[/dim]")
    
    console.print(f"[bold green]✓ Created a total of {len(all_chunks)} chunks from {len(markdown_files)} files![/bold green]")
    return all_chunks, all_metadatas


def initialize_system(docs_dir: str) -> tuple[ChromaDBService, RAGContextProvider]:
    """Initialize the RAG system components with markdown files.
    
    Args:
        docs_dir: Path to the directory containing markdown files
        
    Returns:
        Tuple of (chroma_db, rag_context)
    """
    console.print("\n[bold magenta]🚀 Initializing Markdown RAG Chatbot System...[/bold magenta]")

    try:
        # Load and process markdown files
        markdown_files = load_markdown_files(docs_dir)
        if not markdown_files:
            raise ValueError("No markdown files found to process.")
        
        # Process the markdown files into chunks
        chunks, metadatas = process_markdown_files(markdown_files, CHUNK_SIZE, CHUNK_OVERLAP)
        console.print(f"[dim]• Created {len(chunks)} document chunks[/dim]")

        # Initialize ChromaDB
        console.print("[dim]• Initializing vector database...[/dim]")
        collection_name = f"markdown_docs_{os.path.basename(os.path.normpath(docs_dir))}"
        chroma_db = ChromaDBService(
            collection_name=collection_name, 
            persist_directory=CHROMA_PERSIST_DIR, 
            recreate_collection=True
        )

        # Add chunks to ChromaDB
        console.print("[dim]• Adding document chunks to vector database...[/dim]")
        chunk_ids = chroma_db.add_documents(
            documents=chunks, 
            metadatas=metadatas
        )
        console.print(f"[dim]• Added {len(chunk_ids)} chunks to vector database[/dim]")

        # Initialize context provider
        console.print("[dim]• Creating context provider...[/dim]")
        rag_context = RAGContextProvider("Markdown RAG Context")

        # Register context provider with agents
        console.print("[dim]• Registering context provider with agents...[/dim]")
        query_agent.register_context_provider("rag_context", rag_context)
        qa_agent.register_context_provider("rag_context", rag_context)

        console.print("[bold green]✨ System initialized successfully![/bold green]\n")
        return chroma_db, rag_context

    except Exception as e:
        console.print(f"\n[bold red]Error during initialization:[/bold red] {str(e)}")
        raise


def display_welcome() -> None:
    """Display welcome message and starter questions."""
    welcome_panel = Panel(WELCOME_MESSAGE, title="[bold blue]Markdown RAG Chatbot[/bold blue]", border_style="blue", padding=(1, 2))
    console.print("\n")
    console.print(welcome_panel)

    table = Table(
        show_header=True, header_style="bold cyan", box=box.ROUNDED, title="[bold]Example Questions to Get Started[/bold]"
    )
    table.add_column("№", style="dim", width=4)
    table.add_column("Question", style="green")

    for i, question in enumerate(STARTER_QUESTIONS, 1):
        table.add_row(str(i), question)

    console.print("\n")
    console.print(table)
    console.print("\n" + "─" * 80 + "\n")


def display_chunks(chunks: List[ChunkItem]) -> None:
    """Display the retrieved chunks in a formatted way."""
    console.print("\n[bold cyan]📚 Retrieved Text Chunks:[/bold cyan]")

    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get('file_name', os.path.basename(chunk.metadata.get('source', 'Unknown')))
        chunk_panel = Panel(
            Markdown(chunk.content),
            title=f"[bold]Chunk {i} - Source: {source} (Distance: {chunk.metadata['distance']:.4f})[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(chunk_panel)
        console.print()


def display_query_info(query_output: RAGQueryAgentOutputSchema) -> None:
    """Display information about the generated query."""
    query_panel = Panel(
        f"[yellow]Generated Query:[/yellow] {query_output.query}\n\n" f"[yellow]Reasoning:[/yellow] {query_output.reasoning}",
        title="[bold]🔍 Semantic Search Strategy[/bold]",
        border_style="yellow",
        padding=(1, 2),
    )
    console.print("\n")
    console.print(query_panel)


def display_answer(qa_output: RAGQuestionAnsweringAgentOutputSchema) -> None:
    """Display the reasoning and answer from the QA agent."""
    # Display reasoning
    reasoning_panel = Panel(
        Markdown(qa_output.reasoning),
        title="[bold]🤔 Analysis & Reasoning[/bold]",
        border_style="green",
        padding=(1, 2),
    )
    console.print("\n")
    console.print(reasoning_panel)

    # Display answer
    answer_panel = Panel(
        Markdown(qa_output.answer),
        title="[bold]💡 Answer[/bold]",
        border_style="blue",
        padding=(1, 2),
    )
    console.print("\n")
    console.print(answer_panel)


def chat_loop(chroma_db: ChromaDBService, rag_context: RAGContextProvider) -> None:
    """Main chat loop."""
    display_welcome()

    while True:
        try:
            user_message = console.input("\n[bold blue]Your question:[/bold blue] ").strip()

            if user_message.lower() in ["exit", "quit", "/exit", "/quit"]:
                console.print("\n[bold]👋 Goodbye! Thanks for using the Markdown RAG Chatbot.[/bold]")
                break

            # Check the memory
            if user_message.lower() in ["memory", "mem"]:
                console.print(query_agent.memory.history, style="bold green")
                console.print(qa_agent.memory.history, style="bold green")
                continue


            console.print("\n" + "─" * 80)
            console.print("\n[bold magenta]🔄 Processing your question...[/bold magenta]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Generate search query
                task = progress.add_task("[cyan]Generating semantic search query...", total=None)
                query_output = query_agent.run(RAGQueryAgentInputSchema(user_message=user_message))
                progress.remove_task(task)

                # Display query information
                display_query_info(query_output)

                # Perform vector search
                task = progress.add_task("[cyan]Searching knowledge base...", total=None)
                search_results = chroma_db.query(query_text=query_output.query, n_results=NUM_CHUNKS_TO_RETRIEVE)

                # Update context with retrieved chunks
                rag_context.chunks = [
                    ChunkItem(content=doc, metadata={"chunk_id": id, "distance": dist, "source": meta.get("source", ""), "file_name": meta.get("file_name", "")})
                    for doc, id, dist, meta in zip(
                        search_results["documents"], 
                        search_results["ids"], 
                        search_results["distances"],
                        search_results["metadatas"]
                    )
                ]
                progress.remove_task(task)

                # Display retrieved chunks
                display_chunks(rag_context.chunks)

                # Generate answer
                task = progress.add_task("[cyan]Analyzing chunks and generating answer...", total=None)
                qa_output = qa_agent.run(RAGQuestionAnsweringAgentInputSchema(question=user_message))
                progress.remove_task(task)

                # Display answer
                display_answer(qa_output)

            console.print("\n" + "─" * 80)

        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            console.print("[dim]Please try again or type 'exit' to quit.[/dim]")


def main():
    """Main entry point."""
    console.print("[bold blue]Markdown RAG Chatbot[/bold blue]")
    console.print("[dim]This chatbot uses RAG to answer questions about your markdown documentation.[/dim]")
    
    # Get the docs directory from the user
    docs_dir = console.input("\n[bold yellow]Enter the path to your markdown documentation directory:[/bold yellow] ").strip()
    
    if not docs_dir:
        console.print("[bold red]Error: No directory specified. Using default '../../../docs'[/bold red]")
        docs_dir = "../../../docs"  # Default to the docs directory at the project root
    
    try:
        chroma_db, rag_context = initialize_system(docs_dir)
        chat_loop(chroma_db, rag_context)
    except KeyboardInterrupt:
        console.print("\n[bold]👋 Goodbye! Thanks for using the Markdown RAG Chatbot.[/bold]")
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()