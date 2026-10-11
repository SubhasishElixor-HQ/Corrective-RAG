# Gemini Corrective RAG (CRAG)

A simple Corrective Retrieval-Augmented Generation (CRAG) project using Python, LangChain, LangGraph, Google Gemini, and FAISS.

## 1. Project Workflow

```text
User Question
     |
     v
FAISS Document Retrieval
     |
     v
Gemini Relevance Evaluation
     |
     +---- Relevant ----> Generate Answer
     |
     +---- Irrelevant ---> Correct Search Query
                                |
                                v
                         Retrieve Again
                                |
                                v
                       Evaluate Relevance
                                |
                    +-----------+-----------+
                    |                       |
                Relevant                Irrelevant
                    |                       |
                    v                       v
              Generate Answer        Stop: Insufficient
                                      Evidence
```

## 2. Prerequisites

- Python
- [uv](https://docs.astral.sh/uv/)
- Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## 3. Setup

Open PowerShell:

```powershell

uv add langgraph langchain-community langchain-text-splitters langchain-google-genai pypdf faiss-cpu python-dotenv
```

Create a `.env` file in the project root:

```dotenv
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

Ensure `.env` is included in `.gitignore` to protect your API key.

## 4. Project Structure

```text
RAG-Learning/
├── data/
│   └── business.txt
├── main.py
├── README.md
├── .env
├── .gitignore
├── pyproject.toml
└── uv.lock
```

- `main.py` — CRAG implementation.
- `data/business.txt` — Knowledge document.
- `.env` — Gemini API key.
- `pyproject.toml` and `uv.lock` — Project dependencies.

## 5. Run the Project

```powershell
uv run python main.py
```

Example question:

```text
How does inventory monitoring help a business?
```

The system retrieves relevant document chunks, evaluates their relevance, corrects the query if needed, and generates an answer when sufficient relevant evidence is found.

To exit:

```text
exit
```

## 6. Technologies

- **LangChain:** Document processing, embeddings, and model integration.
- **LangGraph:** CRAG workflow and conditional routing.
- **Google Gemini:** Relevance evaluation, query correction, and answer generation.
- **FAISS:** Vector similarity search.
- **uv:** Python dependency management.

## Note

This is a learning implementation with one corrective retrieval attempt. Keep your API key private, and ensure the Gemini model IDs configured in `main.py` are available to your account.