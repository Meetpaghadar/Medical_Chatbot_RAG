from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from pydantic import Field
from typing import List, Optional
from logger import logger
import os
from dotenv import load_dotenv

router=APIRouter()
load_dotenv()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        # Embed model + Pinecone setup
        pinecone_api_key = (os.getenv("PINECONE_API_KEY") or "").strip()
        if not pinecone_api_key:
            return JSONResponse(status_code=500, content={"error": "Missing PINECONE_API_KEY in .env"})

        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "medicalindex"))
        expected_dimension = 3072
        try:
            desc = pc.describe_index(os.getenv("PINECONE_INDEX_NAME", "medicalindex"))
            index_dimension = getattr(desc, "dimension", None) or desc.get("dimension")
        except Exception:
            index_dimension = None
        if index_dimension is not None and index_dimension != expected_dimension:
            return JSONResponse(
                status_code=500,
                content={
                    "error": (
                        f"Pinecone index dimension mismatch: index={index_dimension}, expected={expected_dimension}. "
                        f"Delete/recreate Pinecone index (or change `PINECONE_INDEX_NAME` to a new one), then re-upload PDFs."
                    )
                },
            )
        embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        embedded_query = embed_model.embed_query(question)
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            ) for match in res["matches"]
        ]

        # If the metadata doesn't include the chunk text, the model will never find answers.
        # This usually happens when PDFs were uploaded before we stored `metadata["text"]`.
        if not any((d.page_content or "").strip() for d in docs):
            return JSONResponse(
                status_code=500,
                content={
                    "error": (
                        "Pinecone metadata is missing chunk text (`metadata['text']`). "
                        "Re-upload your PDFs so the chunk text is stored, then try again."
                    )
                },
            )

        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})
