import streamlit as st

from rag_pipeline import (
    process_documents,
    ask_documents
)

from utils import save_uploaded_file

from config import APP_NAME

st.set_page_config(
    page_title="Advanced RAG",
    layout="wide"
)

with open("styles.css") as f:

    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

with st.sidebar:

    st.title("📂 Upload Files")

    uploaded_files = st.file_uploader(
        "Upload documents",
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

st.title(APP_NAME)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

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

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input(
    "Ask questions..."
)

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