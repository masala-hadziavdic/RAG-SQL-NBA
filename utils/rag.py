# utils/rag.py

import logging
from typing import Tuple, List, Dict

from mistralai.client import Mistral
from mistralai.client.models import UserMessage

from .config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    SEARCH_K,
)

from .vector_store import VectorStoreManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


SYSTEM_PROMPT = """Tu es NBA Analyst AI, un assistant spécialisé dans l'analyse
des données NBA.

Tu dois répondre à la question uniquement à partir du contexte fourni.

Règles :
- Utilise uniquement les informations présentes dans le contexte.
- Ne fabrique pas de statistiques.
- Si le contexte ne permet pas de répondre avec certitude, indique-le clairement.
- Réponds en français.
- Sois précis et concis.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :
"""


class RAGPipeline:
    """Pipeline RAG réutilisable pour Streamlit, FastAPI et évaluation."""

    def __init__(self):
        if not MISTRAL_API_KEY:
            raise ValueError(
                "MISTRAL_API_KEY est absente. "
                "Vérifie ton fichier .env."
            )

        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.model = MODEL_NAME
        self.vector_store = VectorStoreManager()

        if (
            self.vector_store.index is None
            or not self.vector_store.document_chunks
        ):
            raise RuntimeError(
                "L'index FAISS ou les chunks sont absents. "
                "Exécute d'abord : poetry run python indexer.py"
            )

        logging.info(
            "Pipeline RAG initialisé : %s vecteurs, %s chunks",
            self.vector_store.index.ntotal,
            len(self.vector_store.document_chunks),
        )

    def retrieve(self, question: str) -> List[Dict]:
        """Récupère les chunks pertinents pour une question."""

        results = self.vector_store.search(
            question,
            k=SEARCH_K
        )

        logging.info(
            "Question : %s | %s chunks récupérés",
            question,
            len(results)
        )

        return results

    def generate_answer(
        self,
        question: str,
        search_results: List[Dict]
    ) -> str:
        """Génère la réponse Mistral à partir du contexte récupéré."""

        if not search_results:
            context = (
                "Aucun contexte pertinent n'a été trouvé "
                "dans la base documentaire."
            )
        else:
            context = "\n\n---\n\n".join(
                [
                    (
                        f"Source : "
                        f"{result['metadata'].get('source', 'Inconnue')}\n"
                        f"Contenu : {result['text']}"
                    )
                    for result in search_results
                ]
            )

        prompt = SYSTEM_PROMPT.format(
            context=context,
            question=question
        )

        messages = [
            UserMessage(content=prompt)
        ]

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=0.1,
            )

            if not response.choices:
                return (
                    "Aucune réponse valide n'a été générée "
                    "par le modèle."
                )

            answer = response.choices[0].message.content

            if answer is None:
                return "Le modèle n'a retourné aucun contenu."

            return str(answer).strip()

        except Exception as e:
            logging.exception(
                "Erreur lors de la génération de la réponse : %s",
                e
            )
            return (
                "Une erreur est survenue lors de la génération "
                "de la réponse."
            )

    def ask(self, question: str) -> Tuple[str, str]:
        """
        Exécute le pipeline complet :

        question
            ↓
        FAISS retrieval
            ↓
        contexte
            ↓
        Mistral
            ↓
        réponse
        """

        search_results = self.retrieve(question)

        answer = self.generate_answer(
            question,
            search_results
        )

        context = "\n\n".join(
            result["text"]
            for result in search_results
        )

        return answer, context