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


```bash
python index_data.py
```


```bash
python app_qa.py
```

**Sample Output**

Below is an example question and the kind of answer the RAG pipeline is intended to produce when queried against your indexed documents.

- **Q:** What is the NTU's global standing in the QS World University Rankings 2024 for all subjects?

- **A:** According to the provided reports, NTU is ranked among the top 50 universities globally in QS rankings. For a specific year, it was stated that eight of their subjects are in the global top 10 and three interdisciplinary academic subjects - materials science, nanoscience and nanotechnology, and energy and fuels - are ranked the world's best by several independent international entities. However, I do not have information on NTU's exact ranking in the QS World University Rankings 2024 for all subjects.

- **Q:** Summarize the two main takeaways from the section titled "In Conversation with the NTU President" in the 2025 annual report.

- **A:** The two main takeaways from the section titled "In Conversation with the NTU President" are that:

1. The President aims to make NTU a top technological university by achieving significant accomplishments such as ramping up recruitment of top talent, establishing a new college, and boosting community cohesion.
2. As the University moves forward, the President is focused on building on NTU's momentum and realising its ambitions in areas like education, research, innovation, and service, with the ultimate goal of creating a collaborative community that addresses global challenges.

- **Q:** Has the Group's 'Gains in profit or loss' from unquoted investments shown a year-on-year increase from 2023 through 2025?

- **A:** No, it appears that the Group has experienced an overall decrease in gains on unquoted investments. The gain for the university company decreased by $144,369,000 from 2022 to 2023 and then further decreased by $173,077,000 from 2023 to 2024.