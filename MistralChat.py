# MistralChat.py (version RAG + SQL)

import streamlit as st
import os
import logging

from dotenv import load_dotenv
from mistralai.client.models import UserMessage
from mistralai.client import Mistral

from utils.logfire_config import configure_logfire


# ============================================================
# CHARGEMENT .ENV
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION LOGFIRE
# ============================================================

configure_logfire()


# ============================================================
# IMPORTATIONS DEPUIS VOS MODULES
# ============================================================

try:
    from utils.config import (
        MISTRAL_API_KEY,
        MODEL_NAME,
        SEARCH_K,
        APP_TITLE,
        NAME,
    )

    from utils.vector_store import VectorStoreManager
    from database.sql_tool import query_nba_database

except ImportError as e:

    st.error(
        f"Erreur d'importation: {e}. "
        "Vérifiez la structure de vos dossiers et les fichiers dans 'utils'."
    )

    st.stop()
# ============================================================
# CONFIGURATION DU LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)

# ============================================================
# CONFIGURATION DE L'API MISTRAL
# ============================================================

api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error(
        "Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). "
        "Veuillez la définir dans le fichier .env."
    )
    st.stop()

try:
    client = Mistral(api_key=MISTRAL_API_KEY)
    logging.info("Client Mistral initialisé.")

except Exception as e:
    st.error(
        f"Erreur lors de l'initialisation du client Mistral : {e}"
    )
    logging.exception("Erreur initialisation client Mistral")
    st.stop()

# ============================================================
# CHARGEMENT DU VECTOR STORE
# ============================================================

@st.cache_resource
def get_vector_store_manager():

    logging.info(
        "Tentative de chargement du VectorStoreManager..."
    )

    try:
        manager = VectorStoreManager()

        if manager.index is None or not manager.document_chunks:

            st.error(
                "L'index vectoriel ou les chunks n'ont pas pu être chargés."
            )

            st.warning(
                "Assurez-vous d'avoir exécuté "
                "'python indexer.py' après avoir placé vos fichiers "
                "dans le dossier 'inputs'."
            )

            logging.error(
                "Index Faiss ou chunks non trouvés/chargés."
            )

            return None

        logging.info(
            f"VectorStoreManager chargé avec succès "
            f"({manager.index.ntotal} vecteurs)."
        )

        return manager

    except FileNotFoundError:

        st.error(
            "Fichiers d'index ou de chunks non trouvés."
        )

        st.warning(
            "Veuillez exécuter 'python indexer.py' "
            "pour créer la base de connaissances."
        )

        logging.error(
            "FileNotFoundError lors de l'init de VectorStoreManager."
        )

        return None

    except Exception as e:

        st.error(
            f"Erreur inattendue lors du chargement du "
            f"VectorStoreManager: {e}"
        )

        logging.exception(
            "Erreur chargement VectorStoreManager"
        )

        return None

vector_store_manager = get_vector_store_manager()

# ============================================================
# PROMPT SYSTEME RAG
# ============================================================

SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert
sur la ligue de basketball NBA.

Ta mission est de répondre aux questions des fans en utilisant
uniquement les informations fournies dans le contexte.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""

# ============================================================
# PROMPT DU ROUTER
# ============================================================

ROUTER_PROMPT = """Tu es un classificateur de questions pour
un assistant NBA.

Ta tâche est de choisir UNE SEULE source de données :

SQL ou RAG.

Réponds UNIQUEMENT par SQL ou RAG.

IMPORTANT :
La complexité de la question ne détermine PAS le choix.
Une question peut être complexe tout en étant SQL.

============================================================
CHOISIR SQL
============================================================

Choisis SQL lorsque la réponse nécessite des données
numériques, statistiques ou des calculs provenant de la
base de données PostgreSQL.

Exemples :

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
- statistiques sur plusieurs matchs
- statistiques sur une saison
- statistiques récentes

Même si la question est longue ou complexe,
elle doit être classée SQL si la réponse dépend
principalement de données statistiques.

============================================================
CHOISIR RAG
============================================================

Choisis RAG lorsque la réponse nécessite du contenu
textuel, narratif ou contextuel provenant des documents.

Exemples :

- discussions
- analyses
- opinions
- contexte
- histoire
- événements
- explications narratives
- articles
- Reddit
- raisons ou arguments présents dans les documents
- ce que pensent les analystes
- ce que disent les discussions
- informations narratives

============================================================
REGLE ESSENTIELLE
============================================================

Si la question demande des VALEURS NUMERIQUES,
des STATISTIQUES, des CLASSEMENTS ou des CALCULS
à partir des données NBA → SQL.

Si la question demande ce que DISENT LES DOCUMENTS,
les DISCUSSIONS, REDDIT, les ANALYSES ou le CONTEXTE
→ RAG.

============================================================
EXEMPLES SQL
============================================================

"Qui a marqué le plus de points ?" → SQL

"Quel joueur a le plus de rebonds ?" → SQL

"Compare les statistiques de Jokic et Giannis." → SQL

"Combien d'équipes sont dans la base ?" → SQL

"Quels sont les cinq meilleurs scoreurs ?" → SQL

"Quel joueur a le meilleur FG% ?" → SQL

"Quelle équipe a remporté le plus de matchs ?" → SQL

"Quels joueurs ont plus de 20 points par match ?" → SQL

============================================================
EXEMPLES RAG
============================================================

"Que disent les discussions Reddit sur les playoffs ?" → RAG

"Quelles équipes sont considérées comme favorites
et pourquoi ?" → RAG

"Quel contexte explique la rivalité entre ces équipes ?" → RAG

"Que pensent les analystes de cette équipe ?" → RAG

"Que disent les documents sur cette équipe ?" → RAG

============================================================

Question :
{question}

Classification :"""

# ============================================================
# HISTORIQUE DE CONVERSATION
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"Bonjour ! Je suis votre analyste IA pour la "
                f"{NAME}. Posez-moi vos questions sur les équipes, "
                f"les joueurs ou les statistiques."
            )
        }
    ]

# ============================================================
# ROUTER SQL / RAG
# ============================================================

def router_question(question: str) -> str:

    """
    Utilise le LLM pour classifier la question :
    SQL ou RAG.
    """

    try:

        prompt = ROUTER_PROMPT.format(
            question=question
        )

        response = client.chat.complete(
            model=model,
            messages=[
                UserMessage(content=prompt)
            ],
            temperature=0,
        )

        classification = (
            response.choices[0]
            .message.content
            .strip()
            .upper()
        )

        # ----------------------------------------------------
        # NORMALISATION
        # ----------------------------------------------------

        if "SQL" in classification:

            result = "SQL"

        elif "RAG" in classification:

            result = "RAG"

        else:

            result = "RAG"

            logging.warning(
                f"Classification ambiguë '{classification}'. "
                "Utilisation de RAG par défaut."
            )

        logging.info(
            f"Question classifiée comme : {result}"
        )

        return result

    except Exception:

        logging.exception(
            "Erreur lors de la classification"
        )

        return "RAG"

# ============================================================
# GENERATION DE REPONSE
# ============================================================

def generer_reponse(prompt_messages: list) -> str:

    """
    Envoie le prompt à l'API Mistral.
    """

    if not prompt_messages:

        logging.warning(
            "Tentative de génération avec un prompt vide."
        )

        return "Je ne peux pas traiter une demande vide."

    try:

        logging.info(
            f"Appel API Mistral modèle '{model}' "
            f"avec {len(prompt_messages)} message(s)."
        )

        response = client.chat.complete(
            model=model,
            messages=prompt_messages,
            temperature=0.1,
        )

        if response.choices:

            logging.info(
                "Réponse reçue de l'API Mistral."
            )

            return response.choices[0].message.content

        else:

            logging.warning(
                "L'API n'a pas retourné de choix valide."
            )

            return (
                "Désolé, je n'ai pas pu générer "
                "de réponse valide pour le moment."
            )

    except Exception as e:

        st.error(
            f"Erreur lors de l'appel à l'API Mistral : {e}"
        )

        logging.exception(
            "Erreur API Mistral"
        )

        return (
            "Je suis désolé, une erreur technique "
            "m'empêche de répondre. Veuillez réessayer plus tard."
        )

# ============================================================
# PIPELINE SQL
# ============================================================

def handle_sql_question(question: str) -> tuple[str, str]:

    """
    Traite une question via le Tool SQL PostgreSQL.

    Retourne :
        réponse
        source_info
    """

    try:

        result = query_nba_database(question)

        response = result["answer"]

        source_info = (
            "Source: Base de données SQL PostgreSQL\n"
            f"Requête: {result['sql_query']}"
        )

        logging.info(
            f"Réponse SQL générée pour : {question}"
        )

        return response, source_info

    except Exception as e:

        logging.exception(
            "Erreur lors du traitement SQL"
        )

        return (
            f"Erreur lors de la requête SQL : {e}",
            ""
        )

# ============================================================
# PIPELINE RAG
# ============================================================

def handle_rag_question(question: str) -> tuple[str, str]:

    """
    Traite une question via le pipeline RAG.

    Retourne :
        réponse
        source_info
    """

    # --------------------------------------------------------
    # VERIFICATION VECTOR STORE
    # --------------------------------------------------------

    if vector_store_manager is None:

        return (
            "Le service de recherche de connaissances "
            "n'est pas disponible.",
            ""
        )

    # --------------------------------------------------------
    # RECHERCHE VECTORIELLE
    # --------------------------------------------------------

    try:

        search_results = vector_store_manager.search(
            question,
            k=SEARCH_K
        )

        logging.info(
            f"{len(search_results)} chunks trouvés "
            f"pour la question : {question}"
        )

    except Exception:

        logging.exception(
            f"Erreur recherche RAG : {question}"
        )

        search_results = []

    # --------------------------------------------------------
    # CONSTRUCTION DU CONTEXTE
    # --------------------------------------------------------

    context_str = "\n\n---\n\n".join(
        [
            (
                f"Source: "
                f"{res['metadata'].get('source', 'Inconnue')} "
                f"(Score: {res['score']:.1f}%)\n"
                f"Contenu: {res['text']}"
            )
            for res in search_results
        ]
    )

    if not search_results:

        context_str = (
            "Aucune information pertinente trouvée "
            "dans la base de connaissances."
        )

        logging.warning(
            f"Aucun contexte trouvé pour : {question}"
        )

    # --------------------------------------------------------
    # PROMPT RAG
    # --------------------------------------------------------

    final_prompt_for_llm = SYSTEM_PROMPT.format(
        context_str=context_str,
        question=question
    )

    messages_for_api = [
        UserMessage(
            content=final_prompt_for_llm
        )
    ]

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    response_content = generer_reponse(
        messages_for_api
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    if search_results:

        sources = [
            (
                f"- "
                f"{res['metadata'].get('source', 'Inconnue')} "
                f"({res['score']:.0f}%)"
            )
            for res in search_results[:3]
        ]

        source_info = (
            "Sources RAG:\n"
            + "\n".join(sources)
        )

    else:

        source_info = ""

    return response_content, source_info

# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.title(APP_TITLE)

st.caption(
    f"Assistant virtuel pour {NAME} | "
    f"Modèle : {model}"
)

# ============================================================
# AFFICHAGE HISTORIQUE
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

# ============================================================
# UNE SEULE ZONE DE SAISIE
# ============================================================

if prompt := st.chat_input(
    f"Posez votre question sur la {NAME}..."
):

    # --------------------------------------------------------
    # QUESTION UTILISATEUR
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.write(prompt)

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        message_placeholder.text(
            "Analyse de votre question..."
        )

        # ----------------------------------------------------
        # 1. ROUTER
        # ----------------------------------------------------

        question_type = router_question(
            prompt
        )

        logging.info(
            f"Pipeline sélectionné : {question_type}"
        )

        message_placeholder.text(
            f"Recherche en cours ({question_type})..."
        )

        # ----------------------------------------------------
        # 2. UNE SEULE PIPELINE EST EXECUTEE
        # ----------------------------------------------------

        if question_type == "SQL":

            response_content, source_info = (
                handle_sql_question(
                    prompt
                )
            )

        else:

            response_content, source_info = (
                handle_rag_question(
                    prompt
                )
            )

        # ----------------------------------------------------
        # 3. AFFICHER REPONSE
        # ----------------------------------------------------

        message_placeholder.write(
            response_content
        )

        # ----------------------------------------------------
        # 4. AFFICHER SOURCES
        # ----------------------------------------------------

        if source_info:

            with st.expander(
                "Voir les sources"
            ):

                st.code(
                    source_info,
                    language=None
                )

    # --------------------------------------------------------
    # HISTORIQUE REPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_content
        }
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Powered by Mistral AI & Faiss & PostgreSQL | "
    "Data-driven NBA Insights"
)