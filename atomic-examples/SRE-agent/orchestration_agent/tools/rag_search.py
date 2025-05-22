import os
import glob
import json
import yaml
import uuid
import shutil
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass # Added import

import instructor
import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pydantic import Field

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator, SystemPromptContextProviderBase
from atomic_agents.lib.components.agent_memory import AgentMemory

# --- Schemas ---
class RAGSearchToolInputSchema(BaseIOSchema):
    """
    Schema for input to a tool for searching through local documents using RAG.
    Takes a query and returns relevant document chunks along with generated answers.
    """
    query: str = Field(..., description="The question or query to search for in the knowledge base.")

class RAGSearchResultItemSchema(BaseIOSchema):
    """This schema represents a single search result item from the RAG system"""
    content: str = Field(..., description="The content chunk from the document")
    source: str = Field(..., description="The source file of this content chunk")
    distance: float = Field(..., description="Similarity score (lower is better)")
    metadata: Dict = Field(..., description="Additional metadata for the chunk")

class RAGSearchToolOutputSchema(BaseIOSchema):
    """This schema represents the output of the RAG search tool."""
    query: str = Field(..., description="The query used for searching")
    results: List[RAGSearchResultItemSchema] = Field(..., description="List of relevant document chunks")
    answer: str = Field(..., description="Generated answer based on the retrieved chunks")
    reasoning: str = Field(..., description="Explanation of how the answer was derived from the chunks")

# --- Configuration ---
class RAGSearchToolConfig(BaseToolConfig):
    """Configuration for the RAG Search Tool."""
    docs_dir: str = Field("./knowledge_base", description="Directory containing documents to index")
    chunk_size: int = Field(1000, description="Size of each text chunk.")
    chunk_overlap: int = Field(200, description="Overlap between text chunks.")
    num_chunks_to_retrieve: int = Field(5, description="Number of chunks to retrieve for RAG.")
    persist_dir: str = Field("./rag_chroma_db", description="Directory to persist ChromaDB data.")
    collection_name: str = Field("rag_documents", description="ChromaDB collection name.")
    embedding_model_name: str = Field("text-embedding-3-small", description="OpenAI embedding model name.")
    llm_model_name: str = Field("gpt-4o-mini", description="LLM model name for agents.")
    recreate_collection_on_init: bool = Field(True, description="Recreate ChromaDB collection on tool initialization.")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key. If None, attempts to use OPENAI_API_KEY env var.")

# --- RAG Context Provider ---
@dataclass # Made ChunkItem a dataclass
class ChunkItem:
    content: str
    metadata: dict

class RAGContextProvider(SystemPromptContextProviderBase):
    def __init__(self, title: str):
        super().__init__(title=title)
        self.chunks: List[ChunkItem] = []

    def get_info(self) -> str:
        if not self.chunks:
            return "No context chunks available."
        return "\n\n".join(
            [
                f"Chunk {idx}:\nSource: {item.metadata.get('source', 'N/A')}\nContent:\n{item.content}\n{'-' * 20}"
                for idx, item in enumerate(self.chunks, 1)
            ]
        )

# --- ChromaDB Service (adapted from rag_chatbot) ---
class ChromaDBService:
    def __init__(
        self,
        collection_name: str,
        embedding_model_name: str,
        openai_api_key: Optional[str],
        persist_directory: str,
        recreate_collection: bool,
    ):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass via config.")
        
        self.embedding_function = OpenAIEmbeddingFunction(api_key=self.api_key, model_name=embedding_model_name)

        if recreate_collection and os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, str]], ids: Optional[List[str]] = None) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return ids

    def query(self, query_text: str, n_results: int = 5) -> Dict:
        count = self.collection.count()
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]} # Chroma returns list of lists
            
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        # Ensure results are properly unpacked if query_texts was a list of one item
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else [],
        }

# --- RAG Agents (adapted from rag_chatbot) ---
class RAGQueryAgentInputSchema(BaseIOSchema):
    """Input schema for the RAG query agent, taking the user's raw message."""
    user_message: str = Field(..., description="The user's question or message to generate a semantic search query for")

class RAGQueryAgentOutputSchema(BaseIOSchema):
    """Output schema for the RAG query agent, providing the generated query and reasoning."""
    reasoning: str = Field(..., description="The reasoning process leading up to the final query")
    query: str = Field(..., description="The semantic search query to use for retrieving relevant chunks")

class RAGQuestionAnsweringAgentInputSchema(BaseIOSchema):
    """Input schema for the RAG QA agent, taking the user's question."""
    question: str = Field(..., description="The user's question to answer")

class RAGQuestionAnsweringAgentOutputSchema(BaseIOSchema):
    """Output schema for the RAG QA agent, providing the answer and reasoning."""
    reasoning: str = Field(..., description="The reasoning process leading up to the final answer")
    answer: str = Field(..., description="The answer to the user's question based on the retrieved context")

# --- Main Tool & Logic ---
class RAGSearchTool(BaseTool):
    input_schema = RAGSearchToolInputSchema
    output_schema = RAGSearchToolOutputSchema

    def __init__(self, config: RAGSearchToolConfig = RAGSearchToolConfig()):
        super().__init__(config)
        self.config = config
        self.api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass via config.")

        self.chroma_db = ChromaDBService(
            collection_name=config.collection_name,
            embedding_model_name=config.embedding_model_name,
            openai_api_key=self.api_key,
            persist_directory=config.persist_dir,
            recreate_collection=config.recreate_collection_on_init,
        )
        
        self._load_and_index_documents()

        client = instructor.from_openai(openai.OpenAI(api_key=self.api_key))

        self.query_agent = BaseAgent(
            BaseAgentConfig(
                client=client,
                model=config.llm_model_name,
                system_prompt_generator=SystemPromptGenerator(
                    background=[
                        "You are an expert at formulating semantic search queries for RAG systems.",
                        "Your role is to convert user questions into effective semantic search queries that will retrieve the most relevant text chunks.",
                    ],
                    steps=[
                        "1. Analyze the user's question to identify key concepts and information needs.",
                        "2. Reformulate the question into a semantic search query that will match relevant content.",
                        "3. Ensure the query captures the core meaning while being general enough to match similar content.",
                    ],
                    output_instructions=[
                        "Generate a clear, concise semantic search query.",
                        "Focus on key concepts and entities from the user's question.",
                        "Avoid overly specific details that might miss relevant matches.",
                        "Explain your reasoning for the query formulation.",
                    ],
                ),
                input_schema=RAGQueryAgentInputSchema,
                output_schema=RAGQueryAgentOutputSchema,
                memory=AgentMemory(max_messages=5)
            )
        )

        self.rag_context_provider = RAGContextProvider("Retrieved Document Chunks")
        self.qa_agent = BaseAgent(
            BaseAgentConfig(
                client=client,
                model=config.llm_model_name,
                system_prompt_generator=SystemPromptGenerator(
                    background=[
                        "You are an expert at answering questions using retrieved context chunks from a RAG system.",
                        "Your role is to synthesize information from the chunks to provide accurate, well-supported answers.",
                        "You must explain your reasoning process before providing the answer.",
                    ],
                    steps=[
                        "1. Analyze the question and available context chunks.",
                        "2. Identify the most relevant information in the chunks.",
                        "3. Explain how you'll use this information to answer the question.",
                        "4. Synthesize information into a coherent answer.",
                    ],
                    output_instructions=[
                        "First explain your reasoning process clearly.",
                        "Then provide a clear, direct answer based on the context.",
                        "If context is insufficient, state this in your reasoning and answer 'I don't have enough information to answer this question based on the provided documents.'",
                        "Never make up information not present in the chunks.",
                        "Focus on being accurate and concise.",
                    ],
                ),
                input_schema=RAGQuestionAnsweringAgentInputSchema,
                output_schema=RAGQuestionAnsweringAgentOutputSchema,
                memory=AgentMemory(max_messages=5)
            )
        )
        self.qa_agent.register_context_provider("rag_context", self.rag_context_provider)

    def _chunk_text(self, text: str) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        current_size = 0
        for paragraph in paragraphs:
            if current_size + len(paragraph) > self.config.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Overlap
                overlap_text = " ".join(current_chunk.split()[-self.config.chunk_overlap // 5:]) # Approx overlap words
                current_chunk = overlap_text + "\n\n" + paragraph if self.config.chunk_overlap > 0 else paragraph
            else:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
            current_size = len(current_chunk)
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _load_and_process_file(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        content = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith(".json"):
                    content = json.dumps(json.load(f))
                elif file_path.endswith(".yml") or file_path.endswith(".yaml"):
                    content = yaml.dump(yaml.safe_load(f))
                else: # .md, .txt
                    content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return [], []

        if not content:
            return [], []
            
        chunks = self._chunk_text(content)
        metadatas = [{"source": file_path, "file_name": os.path.basename(file_path), "chunk_index": i} for i in range(len(chunks))]
        return chunks, metadatas

    def _load_and_index_documents(self):
        print(f"Loading documents from: {self.config.docs_dir}")
        all_chunks = []
        all_metadatas = []
        
        supported_extensions = ["*.json", "*.md", "*.yml", "*.yaml", "*.txt"]
        file_paths = []
        for ext in supported_extensions:
            file_paths.extend(glob.glob(os.path.join(self.config.docs_dir, "**", ext), recursive=True))

        if not file_paths:
            print(f"No documents found in {self.config.docs_dir} with supported extensions.")
            return

        print(f"Found {len(file_paths)} documents to process.")
        for file_path in file_paths:
            print(f"Processing {file_path}...")
            chunks, metadatas = self._load_and_process_file(file_path)
            all_chunks.extend(chunks)
            all_metadatas.extend(metadatas)
        
        if all_chunks:
            print(f"Adding {len(all_chunks)} chunks to ChromaDB...")
            self.chroma_db.add_documents(documents=all_chunks, metadatas=all_metadatas)
            print("Documents indexed successfully.")
        else:
            print("No chunks to index.")

    def run(self, params: RAGSearchToolInputSchema) -> RAGSearchToolOutputSchema:
        # 1. Generate semantic query
        query_agent_input = RAGQueryAgentInputSchema(user_message=params.query)
        query_output = self.query_agent.run(query_agent_input)
        semantic_query = query_output.query
        print(f"Generated semantic query: {semantic_query}")

        # 2. Retrieve relevant chunks
        search_results = self.chroma_db.query(query_text=semantic_query, n_results=self.config.num_chunks_to_retrieve)
        
        retrieved_chunks_for_context = []
        output_results = []

        if search_results["documents"]:
            for doc, meta, dist_val in zip(search_results["documents"], search_results["metadatas"], search_results["distances"]):
                retrieved_chunks_for_context.append(ChunkItem(content=doc, metadata=meta))
                output_results.append(RAGSearchResultItemSchema(content=doc, source=meta.get("source", "N/A"), distance=dist_val, metadata=meta))
        
        self.rag_context_provider.chunks = retrieved_chunks_for_context
        
        if not retrieved_chunks_for_context:
            print("No relevant chunks found.")
            return RAGSearchToolOutputSchema(
                query=params.query,
                results=[],
                answer="I could not find any relevant information in the documents to answer your question.",
                reasoning="No relevant document chunks were retrieved from the knowledge base for the generated semantic query."
            )

        # 3. Generate answer using QA agent
        qa_agent_input = RAGQuestionAnsweringAgentInputSchema(question=params.query)
        qa_output = self.qa_agent.run(qa_agent_input)

        return RAGSearchToolOutputSchema(
            query=params.query,
            results=output_results,
            answer=qa_output.answer,
            reasoning=qa_output.reasoning
        )

# --- Example Usage ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel

    load_dotenv()
    console = Console()

    # --- Setup for testing ---
    # Create a dummy knowledge base directory
    test_docs_dir = "./temp_knowledge_base"
    os.makedirs(test_docs_dir, exist_ok=True)

    # Create dummy files
    with open(os.path.join(test_docs_dir, "test_doc1.md"), "w") as f:
        f.write("# Markdown Document\n\nThis is a test markdown document about apples.\nApples are a type of fruit.")
    
    with open(os.path.join(test_docs_dir, "test_doc2.txt"), "w") as f:
        f.write("Text Document\n\nThis is a text file about bananas.\nBananas are yellow and curved.")

    with open(os.path.join(test_docs_dir, "test_doc3.json"), "w") as f:
        json.dump({"name": "Orange", "color": "orange", "type": "citrus fruit"}, f)

    with open(os.path.join(test_docs_dir, "test_doc4.yml"), "w") as f:
        yaml.dump({"item": "Grape", "details": {"color": "purple/green", "category": "berry"}}, f)
    
    # Ensure OPENAI_API_KEY is set in your .env file or environment
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error: OPENAI_API_KEY not found. Please set it in your .env file or environment.[/bold red]")
        # Clean up dummy directory
        shutil.rmtree(test_docs_dir)
        exit(1)

    rag_config = RAGSearchToolConfig(
        docs_dir=test_docs_dir,
        persist_dir="./temp_rag_chroma_db", # Use a temporary persist_dir for testing
        recreate_collection_on_init=True 
    )
    
    console.print(Panel(f"Initializing RAGSearchTool with docs_dir: '{test_docs_dir}' and persist_dir: '{rag_config.persist_dir}'", title="[bold blue]RAG Tool Test Setup[/bold blue]"))

    try:
        rag_tool = RAGSearchTool(config=rag_config)
        
        test_queries = [
            "What are apples?",
            "Tell me about bananas.",
            "What color are oranges?",
            "What kind of fruit is a grape?"
        ]

        for user_query in test_queries:
            console.print(Panel(f"User Query: {user_query}", title="[bold yellow]Test Query[/bold yellow]"))
            input_params = RAGSearchToolInputSchema(query=user_query)
            output = rag_tool.run(input_params)

            console.print("\n[bold green]RAG Tool Output:[/bold green]")
            console.print(f"  Query: {output.query}")
            console.print(f"  Answer: {output.answer}")
            console.print(f"  Reasoning: {output.reasoning}")
            console.print("  Retrieved Chunks:")
            if output.results:
                for idx, res_item in enumerate(output.results):
                    console.print(f"    Chunk {idx+1}:")
                    console.print(f"      Source: {res_item.source}")
                    console.print(f"      Distance: {res_item.distance:.4f}")
                    console.print(f"      Content: {res_item.content[:100]}...") # Print snippet
            else:
                console.print("    No chunks retrieved.")
            console.print("-" * 30)

    except Exception as e:
        console.print(f"[bold red]An error occurred during testing: {e}[/bold red]")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy directory and ChromaDB persist directory
        if os.path.exists(test_docs_dir):
            shutil.rmtree(test_docs_dir)
        if os.path.exists(rag_config.persist_dir):
            shutil.rmtree(rag_config.persist_dir)
        console.print(Panel("Cleaned up temporary directories.", title="[bold blue]Test Cleanup[/bold blue]"))