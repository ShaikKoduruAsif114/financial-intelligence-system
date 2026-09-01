import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ingest import DATA_DIR, ingest_documents, parse_document
from rag import research_assistant

load_dotenv()

st.set_page_config(page_title="Financial Intelligence System", page_icon="💼", layout="wide")

st.title("💼 Financial Intelligence System")
st.caption("Use real financial filings, tables, and figure captions as first-class evidence sources.")

DATA_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("Document ingestion")
    uploaded_files = st.file_uploader(
        "Upload real documents or charts",
        type=["pdf", "docx", "doc", "xlsx", "xls", "csv", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = DATA_DIR / uploaded_file.name
            save_path.write_bytes(uploaded_file.getvalue())
        st.success(f"Stored {len(uploaded_files)} file(s) in {DATA_DIR}.")

    if st.button("Ingest documents now"):
        with st.spinner("Parsing PDFs, tables, and figure captions..."):
            chunks = ingest_documents(DATA_DIR)
            if chunks:
                st.success(f"Ingested {len(chunks)} structured chunks.")
            else:
                st.warning("No documents were ingested. Addfiles to data/raw or upload new documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about revenue, margins, risk factors, or strategic highlights:"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching and analyzing the document evidence..."):
            try:
                response = research_assistant(prompt)
                st.markdown(response)
            except Exception as exc:
                response = f"**Error:** {str(exc)}"
                st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
