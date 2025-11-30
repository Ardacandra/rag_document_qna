"""Advanced RAG Query Engine for Knowledge Base Q&A.

This script loads the pre-existing vector index from ChromaDB and initializes 
the full RAG pipeline. It utilizes a **local LLM** (via Ollama) and implements a 
**two-stage retrieval strategy** by configuring a high-K retriever followed by 
a **Reranker**. 
It runs a command-line interface (CLI) to accept user queries, retrieve the
most contextually relevant documents, and synthesize the final answer.
"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.llms.ollama import Ollama  # Using Ollama for local LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

# Load YAML config
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "default.yaml"
config = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except Exception as e:
        print(f"Warning: failed to read config {CONFIG_PATH}: {e}")

def cfg(key, default=None):
    return os.getenv(key, config.get(key, default))

# Define paths and settings (env overrides YAML)
CHROMA_PATH = cfg("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = cfg("COLLECTION_NAME", "advanced_rag_collection")
LLM_MODEL = cfg("LLM_MODEL", "llama3.1:8b")
EMBED_MODEL = cfg("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_MODEL_TYPE = cfg("EMBED_MODEL_TYPE", "local")
RERANK_TOP_N = int(cfg("RERANK_TOP_N", 5))
RETRIEVE_K = int(cfg("RETRIEVE_K", 20))

def initialize_rag_pipeline():
    """Initializes LLM, Reranker, and loads the Index."""
    print("Initializing RAG pipeline...")
    
    # Set LlamaIndex Global Settings (LLM and Embeddings)
    # 1. Local LLM (Ollama)
    Settings.llm = Ollama(model=LLM_MODEL, base_url=cfg("OLLAMA_BASE_URL", "http://localhost:11434"))

    # 2. Local Embedding Model (force local if requested)
    if EMBED_MODEL_TYPE == "local":
        embedder = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        try:
            Settings.embed_model = embedder
        except Exception:
            pass
    else:
        # If not local, still try to set the HuggingFace embedder as default fallback
        embedder = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        try:
            Settings.embed_model = embedder
        except Exception:
            pass

    # 3. Load the Index from ChromaDB (created by index_data.py)
    if chromadb is None:
        raise RuntimeError("chromadb is not installed. Install chromadb to load the persisted index.")

    chroma_db_path = Path(CHROMA_PATH)
    client = chromadb.PersistentClient(path=str(chroma_db_path))
    chroma_collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Load the index using the storage context backed by the Chroma vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    print("Index successfully loaded.")

    # 4. Define Components
    # Retriever (pulls the initial large candidate set)
    retriever = index.as_retriever(similarity_top_k=RETRIEVE_K)

    # Reranker (Post-processor to filter the candidate set)
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=RERANK_TOP_N)
    print(f"Reranker initialized. Final context size: {RERANK_TOP_N} chunks.")

    # 5. Assemble the Advanced Query Engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[reranker], # <-- The Reranking step
        # llm=Settings.llm
    )
    
    return query_engine

def main_cli_app():
    """Simple command-line interface for testing the query engine."""
    query_engine = initialize_rag_pipeline()
    print("\n--- RAG Q&A System Ready ---")
    print(f"Using LLM: {Settings.llm.model}")
    
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        if not query:
            continue
            
        print("Processing query...")
        try:
            response = query_engine.query(query)
            
            print("\n--- Answer ---")
            print(response.response)
            
            print("\n--- Top Sources ---")
            for i, node in enumerate(response.source_nodes):
                print(f"Source {i+1} (Score: {node.score:.4f}): {node.text[:100]}...")
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main_cli_app()