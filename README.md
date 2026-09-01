# 💼 Financial Intelligence System

A production-oriented financial research assistant that ingests real-world business documents and preserves document structure instead of flattening everything into plain text.

## What changed
- 📄 Real document ingestion for PDF, DOCX, XLSX, CSV, TXT, and chart image files
- 📊 Tables are preserved as structured markdown tables instead of being collapsed into paragraphs
- 📈 Figures and chart captions are stored as separate `figure` chunks for explicit analysis
- 🖼️ Chart images are summarized through a vision-analysis layer so the system can reason over the actual graph, not just its caption
- 🔍 Hybrid retrieval still combines dense semantic search with BM25, but it now works on real evidence from business documents
- 🧠 The chunking pipeline keeps `kind`, `source`, `page`, and `section` metadata so answers can cite the exact source and structure

## Features
- Real financial documents, not synthetic text generators
- Structured extraction of text, tables, and figure captions
- ChromaDB vector storage with BM25 fallback indexing
- Hybrid retrieval + reranking
- Streamlit UI for chat and file uploads
- Evaluation hooks for quality metrics

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Create a `.env` file containing your LLM and model configuration as needed:
```env
OPENAI_API_KEY=your_openai_key_here
VISION_MODEL=gpt-4o-mini

# Optional Azure OpenAI configuration
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

# Optional Groq OpenAI-compatible endpoint
GROQ_API_KEY=your_groq_key_here
GROQ_VISION_MODEL=llama-3.2-11b-vision-preview
```

### 3. Prepare real documents
Place actual business documents in `data/raw/` such as:
- PDFs
- DOCX files
- XLSX spreadsheets
- CSV exports
- TXT notes

### 4. Ingest the documents
```bash
python ingest.py
```
This creates the ChromaDB vector store and BM25 index in `vectorstore/`.

### 5. Run the app
```bash
streamlit run app.py
```

## Project structure
```text
├── app.py               # Streamlit chat app + optional file upload support
├── rag.py               # RAG pipeline
├── ingest.py            # Real document ingestion with table/figure extraction
├── evaluate.py          # Evaluation scripts
├── download_dataset.py  # Legacy synthetic generator kept for reference only
├── requirements.txt
├── data/
│   └── raw/             # Put your real documents here
├── vectorstore/         # Generated retrieval index
└── tests/
    └── test_document_ingestion.py
```

## Production notes
- Use OCR-capable pipelines for scanned PDFs and chart images when you need fully automated extracted figure summaries.
- Keep a document metadata registry if you want lineage tracking across reports and tables.
- Add a validation step that checks each file type before indexing to prevent malformed or empty documents from entering the vector store.
- For enterprise use, front this with a document upload API, queue-based ingestion, and versioned storage.
