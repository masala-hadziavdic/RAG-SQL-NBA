"""
schemas.py

Modèles Pydantic utilisés dans le pipeline RAG.
Validation des documents, chunks, recherche, embeddings et SQL.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ==========================================================
# Documents
# ==========================================================

class DocumentMetadata(BaseModel):
    source: str
    filename: str
    category: str
    full_path: str
    sheet: Optional[str] = None


class Document(BaseModel):
    page_content: str = Field(..., min_length=1)
    metadata: DocumentMetadata

    @field_validator("page_content")
    @classmethod
    def validate_content(cls, value):
        if not value.strip():
            raise ValueError("Le document est vide.")
        return value


# ==========================================================
# Chunks
# ==========================================================

class ChunkMetadata(BaseModel):
    source: str
    filename: str
    category: str
    full_path: str
    sheet: Optional[str] = None
    chunk_id_in_doc: int
    start_index: int


class Chunk(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata

    @field_validator("id")
    @classmethod
    def validate_id(cls, value):
        if not value.strip():
            raise ValueError("Identifiant du chunk vide.")
        return value


# ==========================================================
# Embeddings
# ==========================================================

class EmbeddingResult(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int
    count: int

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, value):
        if not value:
            raise ValueError("Aucun embedding.")

        dim = len(value[0])

        if any(len(v) != dim for v in value):
            raise ValueError("Dimensions incohérentes.")

        return value


# ==========================================================
# Recherche FAISS
# ==========================================================

class SearchResult(BaseModel):
    score: float = Field(..., ge=0, le=100)
    raw_score: float
    text: str
    metadata: ChunkMetadata