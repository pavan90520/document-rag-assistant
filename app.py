from dotenv import load_dotenv
import os

load_dotenv(override=True)

import streamlit as st
import os

from utils.loader import load_pdf
from utils.splitter import split_docs
from utils.embeddings import get_embeddings
from utils.vectorstore import build_vectorstore
from utils.rag_chain import build_rag_chain

st.set_page_config(page_title="Document RAG Assistant", layout="wide")

st.title("📄 Document RAG Assistant")
st.write("Upload a PDF and ask questions grounded in its content.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    file_path = os.path.join("data/uploaded_files", uploaded_file.name)
    os.makedirs("data/uploaded_files", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    docs = load_pdf(file_path)
    chunks = split_docs(docs)

    embeddings = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    rag_chain = build_rag_chain(retriever)

    query = st.text_input("Ask a question")

    if query:
        answer = rag_chain.invoke(query)
        st.markdown("### Answer")
        st.write(answer)
