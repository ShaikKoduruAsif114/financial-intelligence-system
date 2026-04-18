import os
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import pickle

import chromadb
from chromadb.utils import embedding_functions

# For Reranking
from sentence_transformers import CrossEncoder
import numpy as np

# For LLM Generation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────
COLLECTION_NAME = 'financial_docs'
VECTORSTORE_PATH = 'vectorstore'
TOP_K_RETRIEVE = 10     # How many docs to retrieve from Chroma
TOP_K_RERANK = 3        # How many docs to pass to the LLM after reranking

# ── 1. VECTOR DATABASE RETRIEVAL ────────────────────────────────
def retrieve_documents(query: str, n_results: int = TOP_K_RETRIEVE) -> List[Dict[str, Any]]:
    '''Retrieve top K documents from ChromaDB based on semantic similarity.'''
    client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    
    # Use the exact same embedding model as in ingest.py
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name='BAAI/bge-small-en-v1.5'
    )
    
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    # Format the results into a cleaner list of dictionaries
    retrieved_docs = []
    
    if results['documents'] and results['documents'][0]:
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        for doc, meta in zip(docs, metadatas):
            retrieved_docs.append({
                'text': doc,
                'metadata': meta
            })
            
    return retrieved_docs

def retrieve_bm25(query: str, n_results: int = TOP_K_RETRIEVE) -> List[Dict[str, Any]]:
    '''Retrieve top K documents using BM25 sparse index.'''
    try:
        with open(f"{VECTORSTORE_PATH}/bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
            bm25 = data['bm25']
            chunks = data['chunks']
    except FileNotFoundError:
        print("[WARNING] BM25 index not found. Run ingest.py first.")
        return []

    tokenized_query = query.lower().split(" ")
    scores = bm25.get_scores(tokenized_query)
    
    # Get top n_results indices
    top_n_indices = np.argsort(scores)[::-1][:n_results]
    
    retrieved_docs = []
    for idx in top_n_indices:
        if scores[idx] > 0:
            chunk = chunks[idx]
            retrieved_docs.append({
                'text': chunk['text'],
                'metadata': {'source': chunk['source'], 'page': chunk['page']}
            })
            
    return retrieved_docs

def hybrid_search(query: str, n_results: int = TOP_K_RETRIEVE) -> List[Dict[str, Any]]:
    '''Combine dense and sparse retrievers using Reciprocal Rank Fusion (RRF).'''
    dense_docs = retrieve_documents(query, n_results)
    sparse_docs = retrieve_bm25(query, n_results)
    
    # RRF Hyperparameter
    k = 60
    rrf_scores = {}
    doc_map = {}
    
    # Helper to process lists
    def process_docs(docs):
        for rank, doc in enumerate(docs):
            # Using text as an identifier (can use chunk_id if available)
            doc_id = doc['text']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
                doc_map[doc_id] = doc
            # RRF formula: 1 / (k + rank)
            rrf_scores[doc_id] += 1 / (k + rank + 1)
            
    process_docs(dense_docs)
    process_docs(sparse_docs)
    
    # Sort by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top K docs
    final_docs = [doc_map[doc_id] for doc_id, score in sorted_docs[:n_results]]
    return final_docs

# ── 2. CROSS-ENCODER RERANKING ──────────────────────────────────
def rerank_documents(query: str, documents: List[Dict[str, Any]], top_k: int = TOP_K_RERANK) -> List[Dict[str, Any]]:
    '''Rerank documents using a highly accurate CrossEncoder model.'''
    if not documents:
        return []
        
    # We use a popular, lightweight cross-encoder model
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    
    # Prepare inputs for the cross-encoder: a list of pairs (query, document)
    pairs = [[query, doc['text']] for doc in documents]
    
    # Predict similarity scores
    scores = model.predict(pairs)
    
    # Attach scores to the documents
    for i, doc in enumerate(documents):
        doc['score'] = scores[i]
        
    # Sort backwards by score (highest first)
    reranked_docs = sorted(documents, key=lambda x: x['score'], reverse=True)
    
    # Return the top_k
    return reranked_docs[:top_k]

# ── 3. LLM GENERATION ───────────────────────────────────────────
def generate_answer(query: str, context_docs: List[Dict[str, Any]]) -> str:
    '''Generate an answer using an LLM, given the query and retrieved context.'''
    
    # Format the context so the LLM can read the text and its source easily
    context_text = ""
    for idx, doc in enumerate(context_docs):
        source = doc['metadata'].get('source', 'Unknown source')
        page = doc['metadata'].get('page', 'Unknown page')
        context_text += f"\n[SOURCE: {source}, PAGE: {page}]\n{doc['text']}\n"

    # Define the core instructions (Prompt) for the LLM
    prompt_template = """You are a Financial Research Assistant. 
Use the following pieces of retrieved financial documents to answer the user's question. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
ALWAYS cite your sources clearly at the end of your answer, referencing the Source File and Page number provided in the context.

Context:
{context}

Question:
{question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # LLM inference. Note: you can use Groq or OpenAI here by changing the model name and API key.
    llm = ChatOpenAI(
        model="llama-3.1-8b-instant", 
        temperature=0,
        base_url="https://api.groq.com/openai/v1"
    )
    
    # Create the generation chain
    chain = prompt | llm | StrOutputParser()
    
    # Generate the response
    print("Generating answer...")
    response = chain.invoke({
        "context": context_text,
        "question": query
    })
    
    return response

# ── MAIN PIPELINE ───────────────────────────────────────────────
def research_assistant(query: str) -> str:
    '''Runs the complete RAG pipeline.'''
    print(f"\n--- Processing Query: '{query}' ---")
    
    # Step 1: Base Retrieval (Hybrid)
    print(f"1. Retrieving top {TOP_K_RETRIEVE} documents using Hybrid Search (BM25 + Dense RRF)...")
    base_docs = hybrid_search(query, TOP_K_RETRIEVE)
    
    # Step 2: Reranking
    print(f"2. Reranking and filtering to top {TOP_K_RERANK} documents...")
    reranked_docs = rerank_documents(query, base_docs)
    
    # Step 3: Generation
    print("3. Generating answer using LLM...\n")
    answer = generate_answer(query, reranked_docs)
    
    return answer

if __name__ == '__main__':
    # Test Query
    test_query = "What are the major risk factors or business highlights mentioned in the document?"
    
    try:
        final_answer = research_assistant(test_query)
        print("\n================ ASSISTANT ANSWER ================\n")
        print(final_answer)
        print("\n==================================================")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        print("Note: If you got an OpenAI error, make sure you have an OPENAI_API_KEY in your .env file, "
              "or swap out ChatOpenAI in the script for an open-source model (like HuggingFace endpoint).")
