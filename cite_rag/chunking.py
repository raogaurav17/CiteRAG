"""
Text chunking and vector storage module.

Handles splitting documents into manageable chunks with overlap,
and storing them in the vector database with metadata for citations.
"""

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from hashlib import md5
import logging

logger = logging.getLogger(__name__)


def process_and_store_text(raw_text: str, vectorstore, cfg) -> int:
    """
    Split text into overlapping chunks and store in vector database.

    Args:
        raw_text: Raw document text to process
        vectorstore: PineconeVectorStore instance for storage
        cfg: Configuration object with chunking parameters

    Returns:
        Number of chunks successfully stored
    """
    # Split text using hierarchical separators to preserve document structure
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(raw_text.strip())
    if not chunks:
        return 0

    texts = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        # Generate deterministic ID based on chunk content for deduplication
        doc_id = md5(chunk.encode()).hexdigest()
        metadata = {
            "source": "user_input",
            "title": "Uploaded / Pasted Text",
            "section": "main",
            "position": i,
            "content": chunk  # Stored for citation display
        }
        texts.append(chunk)
        metadatas.append(metadata)
        ids.append(doc_id)

    # Upsert chunks into Pinecone (updates if ID exists, inserts if new)
    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )

    logger.info(f"Upserted {len(texts)} chunks")
    return len(texts)