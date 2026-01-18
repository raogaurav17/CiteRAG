"""
Document retrieval module with optional reranking.

Implements MMR (Maximal Marginal Relevance) search with optional
Cohere reranking for improved relevance and diversity.
"""

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_pinecone import PineconeVectorStore
import os


def build_retriever(vectorstore: PineconeVectorStore, cfg):
    """
    Build a retriever with optional reranking.

    Args:
        vectorstore: PineconeVectorStore instance for document retrieval
        cfg: Configuration object with retrieval parameters

    Returns:
        LangChain retriever (base or with contextual compression/reranking)
    """
    # Create base retriever using MMR strategy
    # MMR balances relevance and diversity to avoid redundant results
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": cfg.retriever.top_k,
            "lambda_mult": cfg.retriever.mmr_lambda
        }
    )

    # Optionally apply Cohere reranking for higher quality results
    if cfg.retriever.rerank.enabled:
        compressor = CohereRerank(
            model=cfg.retriever.rerank.model,
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            top_n=cfg.retriever.rerank.top_n
        )
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

    return base_retriever