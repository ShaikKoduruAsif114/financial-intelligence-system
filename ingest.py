import fitz # PyMuPDF
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import pickle
from rank_bm25 import BM25Okapi

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────
DATA_DIR = Path('data/raw') # folder with your PDFs
CHUNK_SIZE = 500 # tokens per chunk
CHUNK_OVERLAP = 50 # overlap between chunks
COLLECTION_NAME = 'financial_docs' # ChromaDB collection name

# ── STEP 1: Parse PDF to text ─────────────────────────────────
def parse_pdf(pdf_path: Path) -> list[dict]:
    '''Extract text from each page of a PDF with metadata.'''
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        # Basic cleaning: remove excessive whitespace
        text = ' '.join(text.split())
        if len(text) > 50: # skip nearly empty pages
            pages.append({
                'text': text,
                'source': pdf_path.name,
                'page': page_num + 1
            })
    doc.close()
    return pages

# ── STEP 2: Split into chunks ─────────────────────────────────
def chunk_pages(pages: list[dict]) -> list[dict]:
    '''Split page text into overlapping chunks.'''
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=['\n\n', '\n', '.', ' ']
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page['text'])
        for i, split in enumerate(splits):
            chunks.append({
                'text': split,
                'source': page['source'],
                'page': page['page'],
                'chunk_id': f"{page['source']}_p{page['page']}_c{i}"
            })
    return chunks

# ── STEP 3: Store in ChromaDB ─────────────────────────────────
def store_in_chromadb(chunks: list[dict]):
    '''Store chunks with embeddings in ChromaDB.'''
    client = chromadb.PersistentClient(path='vectorstore')
    # Using sentence-transformers (free, runs locally)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name='BAAI/bge-small-en-v1.5' # fast and good quality
    )
    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )
    
    # Add in batches (ChromaDB handles large batches poorly)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        collection.add(
            documents=[c['text'] for c in batch],
            ids=[c['chunk_id'] for c in batch],
            metadatas=[{'source': c['source'], 'page': c['page']} for c in batch]
        )
        print(f'Stored batch {i//batch_size + 1}, total: {min(i+batch_size, len(chunks))} chunks')
    
    print('Creating BM25 index...')
    tokenized_corpus = [c['text'].lower().split(" ") for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open('vectorstore/bm25_index.pkl', 'wb') as f:
        pickle.dump({'bm25': bm25, 'chunks': chunks}, f)
    print('BM25 index saved.')
    
    return collection

# ── MAIN ──────────────────────────────────────────────────────
def main():
    pdf_files = list(DATA_DIR.glob('*.pdf'))
    print(f'Found {len(pdf_files)} PDFs')
    all_chunks = []
    for pdf_path in pdf_files:
        print(f'Processing: {pdf_path.name}')
        pages = parse_pdf(pdf_path)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)
        print(f'  -> {len(chunks)} chunks')
    print(f'Total chunks: {len(all_chunks)}')
    if not all_chunks:
        print("No chunks to store.")
        return
    print('Storing in ChromaDB...')
    store_in_chromadb(all_chunks)
    print('Done! Vectorstore ready.')

if __name__ == '__main__':
    main()
