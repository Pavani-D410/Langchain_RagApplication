import streamlit as st

from rag_pipeline import (
    process_documents,
    ask_documents
)

from utils import save_uploaded_file

from config import APP_NAME

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Advanced RAG Chatbot",
    layout="wide"
)

# =========================
# Sidebar
# =========================

with st.sidebar:

    st.title("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=[
            "pdf",
            "txt",
            "csv",
            "html",
            "docx",
            "pptx"
        ],
        accept_multiple_files=True
    )

    st.markdown("---")

    st.info(
        "Upload multiple documents and chat with them."
    )

# =========================
# Main Title
# =========================

st.title(APP_NAME)

st.markdown(
    "### Multi-Document AI Assistant using Groq + RAG"
)

# =========================
# Session State
# =========================

if "messages" not in st.session_state:

    st.session_state.messages = []

if "retriever" not in st.session_state:

    st.session_state.retriever = None

# =========================
# Process Documents
# =========================

if uploaded_files:

    file_paths = []

    for uploaded_file in uploaded_files:

        file_path = save_uploaded_file(
            uploaded_file
        )

        file_paths.append(file_path)

    with st.spinner(
        "Processing documents..."
    ):

        st.session_state.retriever = (
            process_documents(file_paths)
        )

    st.success(
        "Documents processed successfully!"
    )

# =========================
# Chat History
# =========================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

# =========================
# Chat Input
# =========================

question = st.chat_input(
    "Ask questions about your documents..."
)

# =========================
# Ask Question
# =========================

if question and st.session_state.retriever:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.write(question)

    with st.spinner("Thinking..."):

        result = ask_documents(
            question,
            st.session_state.retriever
        )

    answer = result["answer"]

    sources = result["sources"]

    final_response = f"""
{answer}

📚 Sources:
{', '.join(sources)}
"""

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_response
        }
    )

    with st.chat_message("assistant"):

        st.write(final_response)

elif question and not st.session_state.retriever:

    st.warning(
        "Please upload documents first."
    )