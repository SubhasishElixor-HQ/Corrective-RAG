
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph


# --------------------------------------------------
# 1. ENVIRONMENT AND CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "business.txt"

load_dotenv(PROJECT_DIR / ".env")

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY is missing. Add your Gemini API key "
        "to the .env file in the project root."
    )


# --------------------------------------------------
# 2. INITIALIZE GEMINI MODELS
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)


# --------------------------------------------------
# 3. SAFELY EXTRACT PLAIN TEXT FROM GEMINI
# --------------------------------------------------

def extract_text(content) -> str:
    """Extract readable text from common Gemini response formats."""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        value = content.get("text", content.get("TEXT", ""))
        return str(value).strip() if value is not None else ""

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                value = item.get("text", item.get("TEXT", ""))

                if value is not None and str(value).strip():
                    parts.append(str(value).strip())

            else:
                # Do not turn metadata objects into search queries.
                continue

        return "\n".join(parts).strip()

    return "" if content is None else str(content).strip()


# --------------------------------------------------
# 4. LOAD DOCUMENTS AND BUILD FAISS VECTOR STORE
# --------------------------------------------------

def build_vector_store():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {DATA_FILE}\n"
            "Create data/business.txt and add your knowledge."
        )

    loader = TextLoader(
        str(DATA_FILE),
        encoding="utf-8",
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise RuntimeError(
            "No document chunks were created. Check business.txt."
        )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    print(f"[SETUP] Loaded {len(documents)} document(s).")
    print(f"[SETUP] Created {len(chunks)} chunk(s).")
    print("[SETUP] FAISS vector store is ready.")

    return vector_store


vector_store = build_vector_store()


# --------------------------------------------------
# 5. DEFINE LANGGRAPH STATE
# --------------------------------------------------

class CRAGState(TypedDict):
    question: str
    search_query: str
    documents: list
    relevant: bool
    attempts: int
    answer: str


# --------------------------------------------------
# 6. RETRIEVE NODE
# --------------------------------------------------

def retrieve(state: CRAGState) -> dict:
    query = state.get("search_query") or state["question"]

    # Ensure FAISS always receives a plain string.
    query = str(query).strip()

    print(f"\n[RETRIEVE] Query: {query}")

    documents = vector_store.similarity_search(
        query,
        k=3,
    )

    print(f"[RETRIEVE] Found {len(documents)} chunk(s).")

    return {
        "search_query": query,
        "documents": documents,
        "relevant": False,
    }


# --------------------------------------------------
# 7. EVALUATE NODE
# --------------------------------------------------

def evaluate(state: CRAGState) -> dict:
    documents = state.get("documents", [])

    if not documents:
        print("[EVALUATE] No documents found.")
        return {"relevant": False}

    context = "\n\n".join(
        document.page_content for document in documents
    )

    prompt = f"""
You are a document relevance evaluator.

Determine whether the supplied context contains information
that directly helps answer the user's question.

Reply with exactly one word:
RELEVANT
or
IRRELEVANT

User question:
{state["question"]}

Retrieved context:
{context}

Decision:
""".strip()

    response = llm.invoke(prompt)
    decision = extract_text(response.content).upper()

    # Use the first word to tolerate surrounding whitespace
    # or brief extra commentary. IRRELEVANT must not match.
    first_word = decision.split()[0].strip(".,:;!?") if decision else ""
    relevant = first_word == "RELEVANT"

    print(f"[EVALUATE] Gemini decision: {decision!r}")
    print(f"[EVALUATE] Evidence relevant: {relevant}")

    return {"relevant": relevant}


# --------------------------------------------------
# 8. CORRECTIVE RETRIEVAL NODE
# --------------------------------------------------

def correct_query(state: CRAGState) -> dict:
    question = state["question"]

    prompt = f"""
Rewrite the user's question as a concise search query
using important keywords.

Return only the search query.
Do not include explanations, labels, or metadata.

User question:
{question}

Search query:
""".strip()

    response = llm.invoke(prompt)
    query = extract_text(response.content)

    # Prevent an empty or unusable correction.
    if not query:
        query = question

    # Limit accidental multi-line responses.
    query = " ".join(query.split())

    print(f"[CORRECT] New query: {query}")

    return {
        "search_query": query,
        "attempts": state.get("attempts", 0) + 1,
    }


# --------------------------------------------------
# 9. GENERATE ANSWER NODE
# --------------------------------------------------

def generate_answer(state: CRAGState) -> dict:
    context = "\n\n".join(
        document.page_content
        for document in state.get("documents", [])
    )

    prompt = f"""
Answer the user's question using only the supplied context.

Rules:
- Do not invent facts.
- If the context does not support a claim, say so.
- Give a clear, direct answer.
- You may organize the answer into short bullet points.

Context:
{context}

User question:
{state["question"]}

Answer:
""".strip()

    response = llm.invoke(prompt)
    answer = extract_text(response.content)

    if not answer:
        answer = (
            "I could not generate a readable answer from the "
            "available context."
        )

    print("[ANSWER] Answer generated.")

    return {"answer": answer}


# --------------------------------------------------
# 10. INSUFFICIENT INFORMATION NODE
# --------------------------------------------------

def insufficient(state: CRAGState) -> dict:
    message = (
        "I could not find sufficiently relevant information "
        "in the knowledge base after the corrective retrieval "
        "attempt. Please add relevant information to "
        "data/business.txt or ask a more specific question."
    )

    print("[ANSWER] Insufficient relevant information.")

    return {"answer": message}


# --------------------------------------------------
# 11. CONDITIONAL ROUTING
# --------------------------------------------------

def route_after_evaluation(state: CRAGState) -> str:
    if state["relevant"]:
        return "answer"

    if state.get("attempts", 0) < 1:
        return "correct"

    return "insufficient"


# --------------------------------------------------
# 12. BUILD THE LANGGRAPH WORKFLOW
# --------------------------------------------------

graph_builder = StateGraph(CRAGState)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("evaluate", evaluate)
graph_builder.add_node("correct", correct_query)
graph_builder.add_node("answer", generate_answer)
graph_builder.add_node("insufficient", insufficient)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "evaluate")

graph_builder.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {
        "answer": "answer",
        "correct": "correct",
        "insufficient": "insufficient",
    },
)

graph_builder.add_edge("correct", "retrieve")
graph_builder.add_edge("answer", END)
graph_builder.add_edge("insufficient", END)

crag_app = graph_builder.compile()


# --------------------------------------------------
# 13. INTERACTIVE QUESTION LOOP
# --------------------------------------------------

def main():
    print("\n" + "=" * 50)
    print(" ELIXOR - Corrective RAG (CRAG)")
    print("=" * 50)
    print("Ask questions about the knowledge base.")
    print("Type 'exit' to quit.")

    while True:
        try:
            question = input("\nAsk a question: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not question:
                print("Please enter a question.")
                continue

            initial_state: CRAGState = {
                "question": question,
                "search_query": question,
                "documents": [],
                "relevant": False,
                "attempts": 0,
                "answer": "",
            }

            result = crag_app.invoke(initial_state)

            print("\n" + "-" * 20 + " ANSWER " + "-" * 20)
            print(result.get("answer", "No answer was generated."))
            print("-" * 48)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as exc:
            print(f"\n[ERROR] {type(exc).__name__}: {exc}")
            print(
                "Check your .env API key, model availability, "
                "internet connection, and dependencies."
            )


if __name__ == "__main__":
    main()
