# 💼 Financial Research Assistant

A RAG (Retrieval-Augmented Generation) system for querying financial documents using **Hybrid Search + Reranking + LLM**.

## Features
- 📄 Parses and ingests PDF financial documents
- 🔍 Hybrid Search: Dense (ChromaDB + BGE embeddings) + Sparse (BM25) with RRF fusion
- 🎯 Cross-Encoder reranking for precision
- 🤖 Groq LLM (llama-3.1-8b-instant) for answer generation
- 📊 Ragas evaluation (Faithfulness + Answer Relevancy)
- 🖥️ Streamlit chat UI

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
Create a `.env` file:
```
OPENAI_API_KEY=your_groq_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```
> Note: `OPENAI_API_KEY` holds your **Groq** API key (Groq uses OpenAI-compatible API).

### 3. Prepare documents
Either generate synthetic PDFs:
```bash
python download_dataset.py
```
Or place your own PDFs in `data/raw/`.

### 4. Ingest documents
```bash
python ingest.py
```
This builds the ChromaDB vector store and BM25 index in `vectorstore/`.

### 5. Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

## Testing

**CLI test:**
```bash
python rag.py
```

**Ragas evaluation:**
```bash
python evaluate.py
```

## Project Structure
```
├── app.py               # Streamlit UI
├── rag.py               # RAG pipeline (hybrid search + reranking + LLM)
├── ingest.py            # PDF ingestion → ChromaDB + BM25
├── evaluate.py          # Ragas evaluation metrics
├── download_dataset.py  # Generate synthetic financial PDFs
├── requirements.txt
└── data/raw/            # Place your PDFs here (gitignored)
```
