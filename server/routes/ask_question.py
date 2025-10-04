from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pydantic import Field
from typing import List, Optional
from logger import logger
import os

router = APIRouter()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        # Embed model + Pinecone setup
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        # Determine index dimension and select matching embedding model
        def _get_index_dimension(name: str) -> int | None:
            try:
                desc = pc.describe_index(name)
                if hasattr(desc, "dimension") and desc.dimension is not None:
                    return int(desc.dimension)
                if isinstance(desc, dict):
                    dim = desc.get("dimension") or (desc.get("spec", {}) or {}).get("dimension")
                    return int(dim) if dim else None
            except Exception:
                return None
            return None

        def _pick_model_for_dim(dim: int) -> tuple[str, dict]:
            encode_kwargs = {"normalize_embeddings": True}
            if dim == 384:
                return "sentence-transformers/all-MiniLM-L6-v2", encode_kwargs
            if dim == 768:
                return "sentence-transformers/all-mpnet-base-v2", encode_kwargs
            if dim == 1024:
                return "intfloat/e5-large-v2", encode_kwargs
            raise ValueError(f"Unsupported Pinecone index dimension {dim}. Update mapping in ask_question.py.")

        index_name = os.environ["PINECONE_INDEX_NAME"]
        dim = _get_index_dimension(index_name) or 384
        model_name, encode_kwargs = _pick_model_for_dim(dim)
        # ✅ Use HuggingFace embeddings and normalize for dotproduct metric
        embed_model = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
        embedded_query = embed_model.embed_query(question)

        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            ) for match in res["matches"]
        ]

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
