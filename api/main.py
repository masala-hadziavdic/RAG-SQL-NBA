# api/main.py

import logging
import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mistralai.client.models import UserMessage
from mistralai.client import Mistral


# ============================================================
# ROOT DU PROJET
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# CONFIGURATION
# ============================================================

from utils.config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    SEARCH_K,
)

from utils.vector_store import VectorStoreManager
from database.sql_tool import query_nba_database


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="NBA RAG + SQL API",
    description="API REST pour le système NBA RAG + SQL",
    version="1.0.0",
)


# ============================================================
# CLIENT MISTRAL
# ============================================================

if not MISTRAL_API_KEY:
    logger.warning("MISTRAL_API_KEY absente.")

client = Mistral(
    api_key=MISTRAL_API_KEY
)

model = MODEL_NAME


# ============================================================
# VECTOR STORE
# ============================================================

logger.info("Chargement du VectorStoreManager...")

try:

    vector_store_manager = VectorStoreManager()

    if (
        vector_store_manager.index is None
        or not vector_store_manager.document_chunks
    ):

        logger.warning(
            "Vector store vide ou non chargé."
        )

    else:

        logger.info(
            "Vector store chargé : %s vecteurs, %s chunks",
            vector_store_manager.index.ntotal,
            len(vector_store_manager.document_chunks),
        )

except Exception:

    logger.exception(
        "Erreur lors du chargement du vector store"
    )

    vector_store_manager = None


# ============================================================
# SCHEMAS API
# ============================================================

class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    route: str
    answer: str
    sources: list[str] = []


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "nba-rag-sql-api",
    }


# ============================================================
# ROUTER PROMPT
# ============================================================

ROUTER_PROMPT = """
Tu es un classificateur de questions pour
un assistant NBA.

Ta tâche est de choisir UNE SEULE source de données :

SQL ou RAG.

Réponds UNIQUEMENT par SQL ou RAG.

Choisis SQL lorsque la réponse nécessite :

- statistiques
- points
- rebonds
- passes décisives
- interceptions
- contres
- FG%
- 3P%
- FT%
- EFG%
- TS%
- victoires
- défaites
- classement
- nombre de joueurs
- nombre d'équipes
- moyennes
- comparaisons statistiques
- TOP N
- statistiques individuelles
- statistiques d'équipe
- calculs provenant de PostgreSQL

Choisis RAG lorsque la réponse nécessite :

- analyses
- opinions
- discussions
- contexte
- histoire
- événements
- articles
- Reddit
- informations narratives
- ce que disent les documents

Question :

{question}

Classification :
"""


# ============================================================
# RAG PROMPT
# ============================================================

SYSTEM_PROMPT = """
Tu es 'NBA Analyst AI', un assistant expert
sur la ligue de basketball NBA.

Ta mission est de répondre aux questions des fans
uniquement à partir des informations fournies
dans le contexte.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:
"""


# ============================================================
# ROUTER
# ============================================================

def router_question(question: str) -> str:

    try:

        prompt = ROUTER_PROMPT.format(
            question=question
        )

        response = client.chat.complete(
            model=model,
            messages=[
                UserMessage(
                    content=prompt
                )
            ],
            temperature=0,
        )

        classification = (
            response.choices[0]
            .message.content
            .strip()
            .upper()
        )

        if "SQL" in classification:
            return "SQL"

        if "RAG" in classification:
            return "RAG"

        logger.warning(
            "Classification ambiguë : %s",
            classification,
        )

        return "RAG"

    except Exception:

        logger.exception(
            "Erreur router"
        )

        return "RAG"


# ============================================================
# PIPELINE SQL
# ============================================================

def handle_sql_question(
    question: str,
) -> tuple[str, list[str]]:

    try:

        result = query_nba_database(
            question
        )

        return (
            result["answer"],
            ["PostgreSQL"],
        )

    except Exception as exc:

        logger.exception(
            "Erreur pipeline SQL"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Erreur SQL : {exc}",
        )


# ============================================================
# PIPELINE RAG
# ============================================================

def handle_rag_question(
    question: str,
) -> tuple[str, list[str]]:

    if vector_store_manager is None:

        raise HTTPException(
            status_code=503,
            detail="Vector store indisponible.",
        )

    try:

        search_results = vector_store_manager.search(
            question,
            k=SEARCH_K,
        )

    except Exception as exc:

        logger.exception(
            "Erreur recherche RAG"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Erreur recherche RAG : {exc}",
        )

    logger.info(
        "%s chunks trouvés pour : %s",
        len(search_results),
        question,
    )

    if search_results:

        context_str = "\n\n---\n\n".join(
            [
                (
                    f"Source: "
                    f"{res['metadata'].get('source', 'Inconnue')}\n"
                    f"Score: {res['score']:.1f}%\n"
                    f"Contenu: {res['text']}"
                )
                for res in search_results
            ]
        )

    else:

        context_str = (
            "Aucune information pertinente "
            "n'a été trouvée dans la base."
        )

    final_prompt = SYSTEM_PROMPT.format(
        context_str=context_str,
        question=question,
    )

    try:

        response = client.chat.complete(
            model=model,
            messages=[
                UserMessage(
                    content=final_prompt
                )
            ],
            temperature=0.1,
        )

        answer = (
            response.choices[0]
            .message.content
            .strip()
        )

    except Exception as exc:

        logger.exception(
            "Erreur génération RAG"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Erreur Mistral : {exc}",
        )

    sources = [
        result["metadata"].get(
            "source",
            "Inconnue",
        )
        for result in search_results[:3]
    ]

    return answer, sources


# ============================================================
# ENDPOINT ASK
# ============================================================

@app.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask_question(
    request: QuestionRequest,
):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide.",
        )

    logger.info(
        "Question reçue : %s",
        question,
    )

    # --------------------------------------------------------
    # ROUTER
    # --------------------------------------------------------

    route = router_question(
        question
    )

    logger.info(
        "Pipeline sélectionné : %s",
        route,
    )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    if route == "SQL":

        answer, sources = handle_sql_question(
            question
        )

    else:

        answer, sources = handle_rag_question(
            question
        )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return QuestionResponse(
        question=question,
        route=route,
        answer=answer,
        sources=sources,
    )