"""
Vector database initialization and management module.

Manages Pinecone vector database setup, index creation, and embedding initialization
using Google's Generative AI embeddings.
"""

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
import time


def get_vectorstore(cfg):
    """
    Initialize or retrieve existing Pinecone vectorstore.

    Creates a new Pinecone index if it doesn't exist, initializes embeddings,
    and returns a PineconeVectorStore instance for document storage and retrieval.

    Args:
        cfg: Configuration object with database and embedding parameters

    Returns:
        Initialized PineconeVectorStore instance

    Raises:
        RuntimeError: If PINECONE_API_KEY environment variable is not set
    """
    # Initialize Pinecone client
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")

    pc = Pinecone(api_key=api_key)
    index_name = cfg.db.index_name

    # Check if index already exists
    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:
        # Create new serverless index with specified configuration
        pc.create_index(
            name=index_name,
            dimension=cfg.db.dimension,
            metric=cfg.db.metric,
            spec=ServerlessSpec(
                cloud=cfg.db.cloud,
                region=cfg.db.region,
            ),
        )

        # Wait for index to be ready before proceeding
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    # Initialize embeddings using Google's Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(
        model=cfg.embedding.model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Connect to Pinecone index with embeddings
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )

    return vectorstore


def clear_database(cfg):
    """
    Clear all documents from the Pinecone vector database.

    Deletes all vectors from the index specified in the configuration.

    Args:
        cfg: Configuration object with database parameters
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")

    pc = Pinecone(api_key=api_key)
    index_name = cfg.db.index_name

    try:
        # Get the index
        index = pc.Index(index_name)
        
        # Delete all vectors from the index
        index.delete(delete_all=True)
        
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to clear database: {str(e)}")