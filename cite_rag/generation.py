"""
Answer generation module using Gemini LLM with citations.

Generates grounded answers based on retrieved documents with inline citations,
token estimation, and cost calculation.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import logging
import os

logger = logging.getLogger(__name__)


def generate_grounded_answer(retriever, query: str, cfg):
    """
    Retrieve relevant documents and generate an answer with citations.

    Args:
        retriever: LangChain retriever (with optional reranking)
        query: User's question
        cfg: Configuration object with LLM parameters

    Returns:
        Tuple of (answer, citations, estimated_tokens, estimated_cost)
    """
    start = time.time()

    try:
        # Retrieve documents using MMR or other retrieval strategy
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} docs for query: {query[:60]}...")

        if not docs:
            return (
                "I could not find relevant information in the provided documents.",
                [],
                0,
                0.0,
            )

        # Format retrieved documents for LLM context with citation markers
        context_lines = []
        citations = []

        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            source_info = (
                f"Source: {doc.metadata.get('source', '?')} • "
                f"Pos: {doc.metadata.get('position', '?')}"
            )

            context_lines.append(f"[{i+1}] {content} ({source_info})")

            citations.append(
                {
                    "number": i + 1,
                    "content": content,
                    "source": doc.metadata.get("source", "user_input"),
                    "position": doc.metadata.get("position", "?"),
                }
            )

        context = "\n\n".join(context_lines)

        # Initialize Gemini LLM
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")

        llm = ChatGoogleGenerativeAI(
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            max_output_tokens=cfg.llm.max_tokens,
            google_api_key=gemini_key,
        )

        # Create prompt that enforces grounded responses with citations
        prompt = ChatPromptTemplate.from_template(
            """You are a precise assistant.
Answer the question using ONLY the provided context.
Do not add external knowledge or assumptions.
If the answer is not present in the context, say so clearly.
Use inline citations like [1], [2] referring to the numbered sources.

Context:
{context}

Question: {query}

Answer:"""
        )

        # Execute LLM chain
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "query": query})

        elapsed = time.time() - start

        # Estimate token usage for cost tracking
        # (Gemini doesn't reliably expose token counts)
        input_chars = len(context) + len(query)
        input_t = input_chars // 4 + 150
        output_t = len(answer) // 4
        total_t = input_t + output_t

        # Gemini free tier has no direct cost
        cost = 0.0

        return answer.strip(), citations, total_t, cost

    except Exception as e:
        logger.exception("Generation failed")
        return f"Error during generation: {str(e)}", [], 0, 0.0
