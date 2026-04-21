"""
rag_engine.py
=============
Core RAG pipeline for Nepal Constitution Chatbot.

Pipeline:
  1. Load the Nepal Constitution PDF
  2. Split into manageable chunks
  3. Build a Chroma vector store (semantic search)
  4. Build a BM25 retriever (keyword search)
  5. Combine both into a Hybrid EnsembleRetriever
  6. Connect to Gemini via LangChain
  7. Expose get_answer(query) for FastAPI to call
"""

import os
from pathlib import Path
import threading
from dotenv import load_dotenv

# Load GOOGLE_API_KEY from .env file explicitly
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever          # from main langchain package
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ── Paths ──────────────────────────────────────────────────────────────────────
# PDF is stored one level up (nepal-chatbot root) next to the backend folder
BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "consituition of nepal.pdf"   # copied here by setup
DB_PATH  = str(BASE_DIR / "chroma_db")              # persisted vector store

# ── Globals (loaded once at startup) ──────────────────────────────────────────
_rag_chain = None   # Will be initialized on first call (lazy init)
_init_lock = threading.Lock()


def _build_rag_chain():
    """
    Build the full RAG pipeline and return the chain.
    This runs once — subsequent calls reuse the cached chain.
    """
    print("🔄 Loading Nepal Constitution PDF...")
    
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment. Please check your .env file.")

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found at {PDF_PATH}. "
            "Make sure 'consituition of nepal.pdf' is in the backend/ folder."
        )

    # ── STEP 1: Load & Split ──────────────────────────────────────────────────
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()
    print(f"   Loaded {len(docs)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    print(f"   Created {len(chunks)} chunks.")

    # ── STEP 2: Vector Store (Semantic Search) ────────────────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Check if the vector store already exists on disk
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print("🔄 Loading existing Chroma vector store from disk...")
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
        )
    else:
        print("🔄 Building Chroma vector store (first run may take a minute)...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH,
        )
    
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ── STEP 3: BM25 (Keyword Search) ────────────────────────────────────────
    print("🔄 Building BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    # ── STEP 4: Hybrid Ensemble ───────────────────────────────────────────────
    # Merges semantic and keyword search results — best of both worlds
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.1, 0.8],   # Slightly favour semantic search
    )

    # ── STEP 5: Gemini LLM ───────────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )

    # ── STEP 6: Prompt ───────────────────────────────────────────────────────
    system_prompt = (
        "You are an expert assistant on the Constitution of Nepal (2015). "
        "Answer the user's question clearly and accurately based ONLY on the "
        "context provided below. If the answer is not found in the context, "
        "say: 'I could not find this in the Nepal Constitution.'\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # ── STEP 7: Assemble Chain ────────────────────────────────────────────────
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(hybrid_retriever, combine_docs_chain)

    print("✅ RAG pipeline ready!")
    return rag_chain


def get_answer(query: str) -> str:
    """
    Public function called by FastAPI.
    Passes the user's query through the RAG chain and returns the answer string.
    """
    global _rag_chain

    # Lazy initialization — build the chain on first request
    if _rag_chain is None:
        with _init_lock:
            # Double-check pattern to avoid redundant builds
            if _rag_chain is None:
                _rag_chain = _build_rag_chain()

    result = _rag_chain.invoke({"input": query})
    return result.get("answer", "Sorry, I could not process your request.")


# ── Stand-alone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = "What are the fundamental rights guaranteed by the Nepal Constitution?"
    print(f"\n📌 Query: {test_query}")
    answer = get_answer(test_query)
    print(f"\n💬 Answer:\n{answer}")
