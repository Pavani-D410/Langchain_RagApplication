import os

from dotenv import load_dotenv

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq



from loaders import load_document

from config import (
    MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS
)

# =========================
# Load Environment Variables
# =========================

load_dotenv()

# =========================
# LLM
# =========================

llm = ChatGroq(
    model_name=MODEL_NAME
)

# =========================
# Embeddings


from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# =========================
# Process Documents
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
# Ask Questions
# =========================

def ask_documents(question, retriever):

    retrieved_docs = retriever.invoke(
        question
    )

    context = "\n\n".join(
        [
            doc.page_content
            for doc in retrieved_docs
        ]
    )

    sources = list(
        set(
            [
                doc.metadata.get(
                    "source",
                    "Unknown"
                )
                for doc in retrieved_docs
            ]
        )
    )

    prompt = f"""
You are an intelligent AI assistant.

Use ONLY the provided context.

If information is not available,
say:

'The information is not available in the uploaded documents.'

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources
    }