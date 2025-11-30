"""Index creation helper for the RAG pipeline.

This script loads configuration from environment variables and `configs/default.yaml`.
It reads files from `DATA_DIR`, chunks them, embeds, and writes vectors into a ChromaDB
collection using LlamaIndex's `VectorStoreIndex`.
"""

import os
from pathlib import Path
import sys
import yaml

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Local project paths
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "default.yaml"

# Load YAML config if present
config = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except Exception as e:
        print(f"Warning: failed to read config {CONFIG_PATH}: {e}")

# Helper to get config from env or yaml
def cfg(key, default=None):
    return os.getenv(key, config.get(key, default))

# Runtime settings (env overrides YAML)
DATA_DIR = cfg("DATA_DIR", "knowledge_base")
CHROMA_PATH = cfg("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = cfg("COLLECTION_NAME", "advanced_rag_collection")
EMBED_MODEL = cfg("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = int(cfg("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(cfg("CHUNK_OVERLAP", 20))

def create_index():
    """Load documents, create a vector store index, and persist to ChromaDB.

    Exits gracefully with helpful messages on failure.
    """
    data_path = Path(DATA_DIR)
    print(f"Starting indexing process from directory: {data_path}")

    if not data_path.exists():
        print(f"Error: data directory '{data_path}' does not exist. Create it and add documents.")
        return

    # 1. LOAD DOCUMENTS
    try:
        documents = SimpleDirectoryReader(input_dir=str(data_path)).load_data()
    except Exception as e:
        print(f"Error loading documents from {data_path}: {e}")
        return

    if not documents:
        print(f"No documents found under {data_path}. Nothing to index.")
        return

    print(f"Loaded {len(documents)} documents.")

    # 2. CONFIGURE COMPONENTS
    print(f"Using embedding model: {EMBED_MODEL} (type={cfg('EMBED_MODEL_TYPE', 'local')})")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    # Force LlamaIndex to use the local embedding model instead of any remote provider
    try:
        Settings.embed_model = embed_model
    except Exception:
        # If Settings can't be set for any reason, continue — we still pass the embedder explicitly
        pass
    node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # 3. CHROMADB SETUP
    if chromadb is None:
        print("Error: 'chromadb' is not installed. Install it with 'pip install chromadb' to persist index.")
        return

    chroma_db_path = Path(CHROMA_PATH)
    chroma_db_path.mkdir(parents=True, exist_ok=True)

    try:
        client = chromadb.PersistentClient(path=str(chroma_db_path))
    except Exception as e:
        print(f"Error initializing chromadb PersistentClient at {chroma_db_path}: {e}")
        return

    try:
        chroma_collection = client.get_or_create_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Error creating/getting chroma collection '{COLLECTION_NAME}': {e}")
        return

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. CREATE AND PERSIST INDEX
    try:
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[node_parser, embed_model],
        )
    except Exception as e:
        print(f"Error creating/persisting index: {e}")
        return

    print("\n--- Indexing Complete ---")
    print(f"Index successfully saved to: {chroma_db_path}")


if __name__ == "__main__":
    create_index()