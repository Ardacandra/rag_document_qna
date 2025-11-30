# RAG Document Q&A
Retrieval-Augmented Generation (RAG) pipeline to provide high-fidelity answers from unstructured internal documents (e.g., PDFs, manuals).

**Setup**

- Create and activate a new environment with a specific Python version:

```bash
conda create -n rag_document_qna python=3.10 -y
conda activate rag_document_qna
```

- Install dependencies via `pip` (after activating the env):

```bash
pip install -r requirements.txt
```

- Install Ollama for your operating system for local LLM setup

- Download a model:

```bash
ollama run llama3.1:8b
```

**How to Run**

- Indexing - run one time for indexing and vector store creation:

```bash
python index_data.py
```

- Querying

```bash
python app_qa.py
```