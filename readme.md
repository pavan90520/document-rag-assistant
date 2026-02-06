# Document RAG Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) application
that allows users to upload PDF documents and ask grounded questions.

## Features
- PDF upload and processing
- Semantic search using FAISS
- Gemini Flash powered question answering
- Hallucination-safe responses

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Google Gemini (gemini-flash-latest)

## How to Run
1. Set `GOOGLE_API_KEY` in `.env`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
