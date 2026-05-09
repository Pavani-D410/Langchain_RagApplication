import os

from dotenv import load_dotenv

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from loaders import load_document

from config import (
    MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS
)

# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

# =========================
# LLM
# =========================

llm = ChatGroq(
    model_name=MODEL_NAME
)

# =========================
# EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name=
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# PROCESS DOCUMENTS
# =========================

def process_documents(file_paths):

    all_docs = []

    for file_path in file_paths:

        documents = load_document(
            file_path
        )

        for doc in documents:

            doc.metadata["source"] = (
                os.path.basename(file_path)
            )

        all_docs.extend(documents)

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    )

    docs = text_splitter.split_documents(
        all_docs
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    vectorstore.save_local(
        "vectorstore"
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K_RESULTS
        }
    )

    return retriever

# =========================
# ASK QUESTIONS
# =========================

def ask_documents(question, retriever):

    retrieved_docs = retriever.invoke(
        question
    )

    # =========================
    # No Documents Retrieved
    # =========================

    if not retrieved_docs:

        return {
            "answer":
            "The information is not available in the uploaded documents.",
            "sources":
            None
        }

    # =========================
    # Context
    # =========================

    context = "\n\n".join(
        [
            doc.page_content
            for doc in retrieved_docs
        ]
    )

    # =========================
    # Best Matching Document
    # =========================

    best_doc = retrieved_docs[0]

    source_name = best_doc.metadata.get(
        "source",
        "Unknown"
    )

    page_number = best_doc.metadata.get(
        "page",
        None
    )

    # =========================
    # Source Formatting
    # =========================

    if page_number is not None:

        source_info = (
            f"{source_name} - "
            f"Page {int(page_number) + 1}"
        )

    else:

        source_info = source_name

    # =========================
    # Prompt
    # =========================

    prompt = f"""
You are an intelligent AI assistant.

STRICT RULES:

1. Answer ONLY from the provided context.

2. If answer is NOT found in context,
reply EXACTLY:

The information is not available in the uploaded documents.

3. Do NOT use outside knowledge.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    # =========================
    # Remove Source if No Answer
    # =========================

    normalized_answer = answer.lower().strip()

    if (
        "information is not available"
        in normalized_answer
    ):

        source_info = None

    return {
        "answer": answer,
        "sources": source_info
    }