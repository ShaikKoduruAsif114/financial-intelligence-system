import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from dotenv import load_dotenv

# Ensure we have our LLM setup correctly for Ragas if needed
# Ragas by default uses OpenAI, which needs OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()

from rag import hybrid_search, rerank_documents, generate_answer

def run_evaluation():
    questions = [
        "What is the overall summary of findings mentioned in the document?",
        "Are there any forward-looking statements?",
        "What are the major risk factors discussed?"
    ]
    
    data = {
        "question": [],
        "answer": [],
        "contexts": []
    }
    
    print("Preparing dataset for Ragas evaluation...")
    for q in questions:
        print(f"Processing question: {q}")
        # Run our custom pipeline
        docs = hybrid_search(q, n_results=10)
        reranked = rerank_documents(q, docs, top_k=3)
        answer = generate_answer(q, reranked)
        
        # Ragas needs contexts as a list of strings
        contexts = [d['text'] for d in reranked]
        
        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        
    dataset = Dataset.from_dict(data)
    
    print("\nStarting Ragas Evaluation (Faithfulness & Answer Relevancy)...")
    try:
        # Groq does not support generating multiple outputs per prompt (n>1), 
        # which Ragas requires for certain metrics. We wrap ChatOpenAI to intercept `n`
        # and duplicate the single response to satisfy Ragas without crashing.
        class GroqSafeChatOpenAI(ChatOpenAI):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                n = kwargs.pop("n", 1)
                kwargs["n"] = 1
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                if n > 1:
                    result.generations = result.generations * n
                return result

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                n = kwargs.pop("n", 1)
                kwargs["n"] = 1
                result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                if n > 1:
                    result.generations = result.generations * n
                return result

        # Create LLM pointing to Groq just like in rag.py
        llm = GroqSafeChatOpenAI(
            model="llama-3.1-8b-instant", 
            temperature=0,
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Create Embeddings using the same HuggingFace model as ChromaDB
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            hf_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # In newer ragas versions, they prefer wrapping Langchain models
        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            eval_llm = LangchainLLMWrapper(llm)
            eval_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        except ImportError:
            # Fallback for older ragas versions
            eval_llm = llm
            eval_embeddings = hf_embeddings
        
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
            ],
            llm=eval_llm,
            embeddings=eval_embeddings,
            raise_exceptions=False 
        )
        
        print("\n================ RAGAS SCORES ================\n")
        print(result)
        print("\n==============================================")
        
    except Exception as e:
        print(f"\n[ERROR] Ragas evaluation failed: {e}")
        print("Note: Ragas usually requires an OPENAI_API_KEY. Make sure it's in your .env.")

if __name__ == "__main__":
    run_evaluation()
