"""
evaluate_ragas.py
=================

Évaluation du système hybride RAG + SQL avec RAGAS 0.4.3.

Architecture :

    Question
        |
        v
    ROUTER LLM
        |
        +----------+
        |          |
        v          v
       SQL        RAG
        |          |
        v          v
 query_nba_     FAISS
 database()
        |          |
        +-----+----+
              |
              v
          Réponse
              |
              v
            RAGAS

RAGAS évalue donc les réponses provenant des deux pipelines :
    - SQL
    - RAG

Pour SQL, retrieved_contexts contient :
    - la requête SQL
    - le résultat SQL

Pour RAG, retrieved_contexts contient :
    - les chunks récupérés par FAISS

Métriques :
    - Faithfulness
    - AnswerRelevancy
    - ContextPrecision
    - ContextRecall

Usage :

    poetry run python evaluate_ragas.py

ou :

    poetry run python evaluate_ragas.py --mode hybrid

Pour tester uniquement RAG :

    poetry run python evaluate_ragas.py --mode rag
"""

# ============================================================
# IMPORTS
# ============================================================

import argparse
import json
import logging
import os
import time
import warnings

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import logfire
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from mistralai.client import Mistral
from mistralai.client.models import UserMessage
from openai import OpenAI

from langchain_mistralai import MistralAIEmbeddings

from ragas import evaluate

from ragas.dataset_schema import (
    EvaluationDataset,
    SingleTurnSample,
)

from ragas.embeddings import (
    LangchainEmbeddingsWrapper,
)

from ragas.llms import llm_factory

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# ============================================================
# TES MODULES
# ============================================================

from utils.config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    SEARCH_K,
)

from utils.vector_store import (
    VectorStoreManager,
)

from database.sql_tool import (
    query_nba_database,
)

# ============================================================
# ENVIRONNEMENT
# ============================================================

load_dotenv()

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="ragas",
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

TEST_QUESTIONS_FILE = (
    Path("data/evaluation_questions.json")
)

RESULTS_DIR = Path(
    "data/evaluation_results"
)

MISTRAL_BASE_URL = (
    "https://api.mistral.ai/v1"
)

RAGAS_MODEL = (
    "mistral-small-latest"
)

EMBEDDING_MODEL = (
    "mistral-embed"
)

DELAY_BETWEEN_CALLS = 1


# ============================================================
# PROMPT RAG
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """
Tu es 'NBA Analyst AI', un assistant expert
sur la ligue de basketball NBA.

Ta mission est de répondre aux questions des fans
en utilisant uniquement les informations fournies
dans le contexte.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:
"""


# ============================================================
# PROMPT ROUTER
# ============================================================

ROUTER_PROMPT = """
Tu es un classificateur de questions pour
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
numériques, statistiques ou des calculs provenant de
la base de données NBA.

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
QUESTION
============================================================

{question}

Classification :
"""


# ============================================================
# CHARGEMENT QUESTIONS
# ============================================================

def load_test_questions(
    filepath: Path = TEST_QUESTIONS_FILE,
) -> List[Dict[str, Any]]:
    """
    Charge les questions de test.
    """

    logger.info(
        "Chargement des questions depuis %s...",
        filepath,
    )

    if not filepath.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {filepath}"
        )

    with open(
        filepath,
        "r",
        encoding="utf-8",
    ) as f:

        questions = json.load(f)

    if not isinstance(
        questions,
        list,
    ):
        raise ValueError(
            "Le fichier doit contenir une liste JSON."
        )

    if not questions:
        raise ValueError(
            "Le fichier de questions est vide."
        )

    for i, question in enumerate(
        questions,
        start=1,
    ):

        if not isinstance(
            question,
            dict,
        ):
            raise ValueError(
                f"Question {i} invalide."
            )

        if "question" not in question:
            raise ValueError(
                f"Question {i} : clé 'question' absente."
            )

        if not str(
            question["question"]
        ).strip():

            raise ValueError(
                f"Question {i} vide."
            )

    logger.info(
        "%d questions chargées.",
        len(questions),
    )

    return questions


# ============================================================
# ROUTER LLM
# ============================================================

def router_question(
    question: str,
    mistral_client: Mistral,
) -> str:
    """
    Utilise le même principe de router LLM
    que MistralChat.py.

    Retour :
        SQL
        RAG
    """

    with logfire.span(
        "router-llm",
        question=question[:100],
    ):

        try:

            prompt = ROUTER_PROMPT.format(
                question=question
            )

            response = (
                mistral_client.chat.complete(
                    model=MODEL_NAME,
                    messages=[
                        UserMessage(
                            content=prompt
                        )
                    ],
                    temperature=0,
                )
            )

            classification = (
                response
                .choices[0]
                .message
                .content
                .strip()
                .upper()
            )

            # --------------------------------------------
            # NORMALISATION
            # --------------------------------------------

            if "SQL" in classification:

                pipeline_type = "SQL"

            elif "RAG" in classification:

                pipeline_type = "RAG"

            else:

                pipeline_type = "RAG"

                logger.warning(
                    "Classification ambiguë '%s'. "
                    "RAG utilisé par défaut.",
                    classification,
                )

            logfire.info(
                "Question classifiée",
                classification=pipeline_type,
            )

            logger.info(
                "Router : %s -> %s",
                question,
                pipeline_type,
            )

            return pipeline_type

        except Exception as e:

            logger.exception(
                "Erreur pendant le routing."
            )

            logfire.error(
                "Erreur router",
                erreur=str(e),
            )

            # Même comportement que MistralChat.py
            return "RAG"


# ============================================================
# PIPELINE RAG
# ============================================================

def run_rag_pipeline(
    question: str,
    vector_store: VectorStoreManager,
    mistral_client: Mistral,
) -> Dict[str, Any]:
    """
    Pipeline RAG :

        question
            ↓
        FAISS
            ↓
        contexte
            ↓
        Mistral
            ↓
        réponse
    """

    with logfire.span(
        "pipeline-rag",
        question=question[:100],
    ):

        logger.info(
            "[RAG] Question : %s",
            question,
        )

        # ==================================================
        # 1. RECHERCHE FAISS
        # ==================================================

        try:

            search_results = (
                vector_store.search(
                    question,
                    k=SEARCH_K,
                )
            )

        except Exception as e:

            logger.exception(
                "[RAG] Erreur recherche FAISS."
            )

            logfire.error(
                "Erreur recherche RAG",
                erreur=str(e),
            )

            search_results = []

        logger.info(
            "[RAG] %d chunks trouvés.",
            len(search_results),
        )

        # ==================================================
        # 2. CONSTRUCTION CONTEXTE
        # ==================================================

        if search_results:

            context_str = "\n\n---\n\n".join(
                [
                    (
                        f"Source: {res['metadata'].get('source', 'Source inconnue')} "
                        f"(Score: {res['score']:.1f}%)\n"
                        f"Contenu: {res['text']}"
                    )
                    for res in search_results
                ]
            )

        else:

            context_str = (
                "Aucune information pertinente "
                "trouvée dans la base de connaissances."
            )

        # ==================================================
        # 3. CONSTRUCTION DU PROMPT
        # ==================================================

        final_prompt = (
            SYSTEM_PROMPT_TEMPLATE.format(
                context_str=context_str,
                question=question
            )
        )
        # ==================================================
        # 4. APPEL MISTRAL
        # ==================================================

        with logfire.span(
            "appel-llm",
            modele=MODEL_NAME,
        ):

            response = (
                mistral_client.chat.complete(
                    model=MODEL_NAME,
                    messages=[
                        UserMessage(
                            content=final_prompt
                        )
                    ],
                    temperature=0.1,
                )
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

        # ==================================================
        # 5. CONTEXTES POUR RAGAS
        # ==================================================

        retrieved_contexts = [
            res["text"]
            for res in search_results
        ]

        logfire.info(
            "Pipeline RAG terminé",
            nb_contextes=len(
                retrieved_contexts
            ),
        )

        return {
            "question": question,
            "response": answer,
            "retrieved_contexts": retrieved_contexts,
            "pipeline_type": "RAG",
        }


# ============================================================
# PIPELINE SQL
# ============================================================

def run_sql_pipeline(
    question: str,
) -> Dict[str, Any]:
    """
    Pipeline SQL.

    IMPORTANT :
    query_nba_database() peut utiliser PostgreSQL.
    RAGAS ne dépend pas du type de base de données.

    Le résultat SQL est transformé en contexte
    pour permettre à RAGAS d'évaluer la réponse.
    """

    with logfire.span(
        "pipeline-sql",
        question=question[:100],
    ):

        logger.info(
            "[SQL] Question : %s",
            question,
        )

        try:

            # ==========================================
            # APPEL À TON SQL TOOL POSTGRESQL
            # ==========================================

            result = query_nba_database(
                question
            )

            if not isinstance(
                result,
                dict,
            ):
                raise ValueError(
                    "query_nba_database() doit retourner "
                    "un dictionnaire."
                )

            answer = str(
                result.get(
                    "answer",
                    "",
                )
            ).strip()

            sql_query = str(
                result.get(
                    "sql_query",
                    "",
                )
            ).strip()

            sql_result = result.get(
                "sql_result",
                [],
            )

            # ==========================================
            # CONTEXTE SQL POUR RAGAS
            # ==========================================

            sql_context = (
                "REQUÊTE SQL:\n"
                f"{sql_query}\n\n"
                "RÉSULTAT SQL:\n"
                f"{sql_result}"
            )

            retrieved_contexts = [
                sql_context
            ]

            logfire.info(
                "Pipeline SQL terminé"
            )

            logger.info(
                "[SQL] ✓ Réponse générée."
            )

            return {
                "question": question,
                "response": answer,
                "retrieved_contexts": retrieved_contexts,
                "pipeline_type": "SQL",
                "sql_query": sql_query,
                "sql_result": sql_result,
            }

        except Exception as e:

            logger.exception(
                "[SQL] Erreur SQL."
            )

            logfire.error(
                "Erreur pipeline SQL",
                erreur=str(e),
            )

            return {
                "question": question,
                "response": "",
                "retrieved_contexts": [],
                "pipeline_type": "SQL",
                "error": str(e),
            }


# ============================================================
# PIPELINE HYBRIDE
# ============================================================

def run_hybrid_pipeline(
    question: str,
    vector_store: VectorStoreManager,
    mistral_client: Mistral,
) -> Dict[str, Any]:
    """
    Pipeline hybride :

        Question
            ↓
        Router LLM
            ↓
        SQL OU RAG
    """

    with logfire.span(
        "pipeline-hybrid",
        question=question[:100],
    ):

        # ==================================================
        # 1. ROUTER
        # ==================================================

        pipeline_type = router_question(
            question,
            mistral_client,
        )

        # ==================================================
        # 2. PIPELINE CHOISI
        # ==================================================

        if pipeline_type == "SQL":

            result = run_sql_pipeline(
                question
            )

        else:

            result = run_rag_pipeline(
                question,
                vector_store,
                mistral_client,
            )

        # ==================================================
        # 3. LOG
        # ==================================================

        logfire.info(
            "Pipeline hybride terminé",
            pipeline_type=pipeline_type,
        )

        return result


# ============================================================
# EXECUTION DES QUESTIONS
# ============================================================

def run_all_questions(
    questions: List[Dict[str, Any]],
    vector_store: VectorStoreManager,
    mistral_client: Mistral,
    mode: str = "hybrid",
) -> List[Dict[str, Any]]:
    """
    Exécute toutes les questions.

    mode = hybrid :
        Router -> SQL ou RAG

    mode = rag :
        RAG uniquement
    """

    results = []

    total = len(
        questions
    )

    logger.info("=" * 70)
    logger.info(
        "EXECUTION DU SYSTEME %s",
        mode.upper(),
    )
    logger.info("=" * 70)

    for i, item in enumerate(
        questions,
        start=1,
    ):

        question = str(
            item["question"]
        ).strip()

        logger.info(
            "[%d/%d] %s",
            i,
            total,
            question,
        )

        try:

            # ==============================================
            # HYBRID
            # ==============================================

            if mode == "hybrid":

                result = (
                    run_hybrid_pipeline(
                        question,
                        vector_store,
                        mistral_client,
                    )
                )

            # ==============================================
            # RAG UNIQUEMENT
            # ==============================================

            else:

                result = (
                    run_rag_pipeline(
                        question,
                        vector_store,
                        mistral_client,
                    )
                )

            # ==============================================
            # METADATA
            # ==============================================

            result["category"] = item.get(
                "category",
                "unknown",
            )

            result["reference"] = item.get(
                "reference"
            )

            results.append(
                result
            )

            logger.info(
                "[%d/%d] ✓ %s",
                i,
                total,
                result.get(
                    "pipeline_type",
                    "UNKNOWN",
                ),
            )

        except Exception as e:

            logger.exception(
                "[%d/%d] Erreur.",
                i,
                total,
            )

            results.append(
                {
                    "question": question,
                    "response": "",
                    "retrieved_contexts": [],
                    "pipeline_type": "ERROR",
                    "category": item.get(
                        "category",
                        "unknown",
                    ),
                    "reference": item.get(
                        "reference"
                    ),
                    "error": str(e),
                }
            )

        # ==============================================
        # DELAI API
        # ==============================================

        if i < total:

            time.sleep(
                DELAY_BETWEEN_CALLS
            )

    # =====================================================
    # STATISTIQUES
    # =====================================================

    succeeded = sum(
        1
        for result in results
        if result.get("response")
    )

    sql_count = sum(
        1
        for result in results
        if result.get(
            "pipeline_type"
        ) == "SQL"
    )

    rag_count = sum(
        1
        for result in results
        if result.get(
            "pipeline_type"
        ) == "RAG"
    )

    logger.info("=" * 70)

    logger.info(
        "PIPELINE TERMINE"
    )

    logger.info(
        "Questions : %d",
        total,
    )

    logger.info(
        "Réussies : %d",
        succeeded,
    )

    logger.info(
        "SQL : %d",
        sql_count,
    )

    logger.info(
        "RAG : %d",
        rag_count,
    )

    logger.info("=" * 70)

    return results


# ============================================================
# DATASET RAGAS
# ============================================================

def build_evaluation_dataset(
    results: List[Dict[str, Any]],
) -> EvaluationDataset:
    """
    Construit UN SEUL dataset RAGAS
    contenant SQL + RAG.
    """

    samples = []

    for result in results:

        # On ignore uniquement les erreurs
        if not result.get(
            "response"
        ):
            continue

        sample = SingleTurnSample(
            user_input=result[
                "question"
            ],

            response=result[
                "response"
            ],

            retrieved_contexts=result.get(
                "retrieved_contexts",
                [],
            ),

            reference=result.get(
                "reference"
            ),
        )

        samples.append(
            sample
        )

    if not samples:

        raise RuntimeError(
            "Aucun résultat valide pour RAGAS."
        )

    logger.info(
        "Dataset RAGAS : %d échantillons.",
        len(samples),
    )

    return EvaluationDataset(
        samples=samples
    )


# ============================================================
# LLM RAGAS
# ============================================================

def create_ragas_llm():
    """
    Crée le LLM utilisé par RAGAS.
    """

    if not MISTRAL_API_KEY:

        raise RuntimeError(
            "MISTRAL_API_KEY absente."
        )

    client = OpenAI(
        api_key=MISTRAL_API_KEY,
        base_url=MISTRAL_BASE_URL,
    )

    evaluator_llm = llm_factory(
        RAGAS_MODEL,
        provider="openai",
        client=client,
    )

    logger.info(
        "✓ LLM RAGAS : %s",
        RAGAS_MODEL,
    )

    return evaluator_llm


# ============================================================
# EMBEDDINGS RAGAS
# ============================================================

def create_ragas_embeddings():
    """
    Crée les embeddings utilisés par RAGAS.
    """

    if not MISTRAL_API_KEY:

        raise RuntimeError(
            "MISTRAL_API_KEY absente."
        )

    embeddings = MistralAIEmbeddings(
        api_key=MISTRAL_API_KEY,
        model=EMBEDDING_MODEL,
    )

    ragas_embeddings = (
        LangchainEmbeddingsWrapper(
            embeddings
        )
    )

    logger.info(
        "✓ Embeddings RAGAS : %s",
        EMBEDDING_MODEL,
    )

    return ragas_embeddings


# ============================================================
# METRIQUES
# ============================================================

def create_metrics(
    evaluator_llm,
    evaluator_embeddings,
) -> Dict[str, Any]:
    """
    Initialise les quatre métriques RAGAS.
    """

    metrics = {

        "faithfulness": Faithfulness(
            llm=evaluator_llm
        ),

        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),

        "context_precision": ContextPrecision(
            llm=evaluator_llm
        ),

        "context_recall": ContextRecall(
            llm=evaluator_llm
        ),
    }

    logger.info(
        "✓ 4 métriques RAGAS initialisées."
    )

    return metrics


# ============================================================
# EVALUATION RAGAS
# ============================================================

def run_evaluation(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    evaluator_llm,
    evaluator_embeddings,
) -> pd.DataFrame:
    """
    Évalue SQL + RAG.

    Pass 1 :
        Faithfulness
        AnswerRelevancy

    Pass 2 :
        ContextPrecision
        ContextRecall

    Les résultats SQL et RAG restent
    dans le même dataset global.
    """

    # ========================================================
    # DATASET GLOBAL
    # ========================================================

    dataset_all = (
        build_evaluation_dataset(
            results
        )
    )

    # ========================================================
    # PASS 1
    # ========================================================

    logger.info("=" * 70)
    logger.info(
        "PASS 1 : FAITHFULNESS + ANSWER RELEVANCY"
    )
    logger.info(
        "SQL + RAG"
    )
    logger.info("=" * 70)

    result_pass_1 = evaluate(
        dataset=dataset_all,
        metrics=[
            metrics[
                "faithfulness"
            ],
            metrics[
                "answer_relevancy"
            ],
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    df_scores = (
        result_pass_1.to_pandas()
    )

    logger.info(
        "✓ Pass 1 terminée."
    )

    # ========================================================
    # INITIALISATION PASS 2
    # ========================================================

    df_scores[
        "context_precision"
    ] = np.nan

    df_scores[
        "context_recall"
    ] = np.nan

    # ========================================================
    # QUESTIONS AVEC REFERENCE
    # ========================================================

    results_with_ref = [
        result
        for result in results
        if (
            result.get("response")
            and result.get("reference")
        )
    ]

    if not results_with_ref:

        logger.warning(
            "Aucune référence disponible "
            "pour ContextPrecision / ContextRecall."
        )

        return df_scores

    # ========================================================
    # PASS 2
    # ========================================================

    logger.info("=" * 70)
    logger.info(
        "PASS 2 : CONTEXT PRECISION + CONTEXT RECALL"
    )
    logger.info(
        "SQL + RAG AVEC REFERENCE"
    )
    logger.info("=" * 70)

    dataset_ref = (
        build_evaluation_dataset(
            results_with_ref
        )
    )

    result_pass_2 = evaluate(
        dataset=dataset_ref,
        metrics=[
            metrics[
                "context_precision"
            ],
            metrics[
                "context_recall"
            ],
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    df_ref = (
        result_pass_2.to_pandas()
    )

    # ========================================================
    # FUSION PAR QUESTION
    # ========================================================

    ref_questions = [
        result["question"]
        for result in results_with_ref
    ]

    for idx, row in df_ref.iterrows():

        if idx >= len(
            ref_questions
        ):
            continue

        question = (
            ref_questions[idx]
        )

        mask = (
            df_scores[
                "user_input"
            ]
            == question
        )

        if not mask.any():
            continue

        if (
            "context_precision"
            in row
        ):

            df_scores.loc[
                mask,
                "context_precision",
            ] = row[
                "context_precision"
            ]

        if (
            "context_recall"
            in row
        ):

            df_scores.loc[
                mask,
                "context_recall",
            ] = row[
                "context_recall"
            ]

    logger.info(
        "✓ Pass 2 terminée."
    )

    return df_scores


# ============================================================
# AJOUT METADATA
# ============================================================

def add_metadata(
    df_scores: pd.DataFrame,
    results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Ajoute category et pipeline_type.
    """

    metadata = {
        result["question"]: {
            "category": result.get(
                "category",
                "unknown",
            ),
            "pipeline_type": result.get(
                "pipeline_type",
                "unknown",
            ),
        }
        for result in results
    }

    df_scores[
        "category"
    ] = (
        df_scores[
            "user_input"
        ]
        .map(
            lambda question:
                metadata.get(
                    question,
                    {},
                ).get(
                    "category",
                    "unknown",
                )
        )
    )

    df_scores[
        "pipeline_type"
    ] = (
        df_scores[
            "user_input"
        ]
        .map(
            lambda question:
                metadata.get(
                    question,
                    {},
                ).get(
                    "pipeline_type",
                    "unknown",
                )
        )
    )

    return df_scores


# ============================================================
# AGREGATS
# ============================================================

def compute_aggregate_scores(
    df: pd.DataFrame,
    metric_columns: List[str],
) -> pd.DataFrame:
    """
    Moyennes par catégorie + global.
    """

    if not metric_columns:

        return pd.DataFrame()

    numeric_df = df.copy()

    for column in metric_columns:

        numeric_df[
            column
        ] = pd.to_numeric(
            numeric_df[
                column
            ],
            errors="coerce",
        )

    aggregate = (
        numeric_df
        .groupby(
            "category"
        )[metric_columns]
        .mean()
    )

    aggregate.loc[
        "all"
    ] = (
        numeric_df[
            metric_columns
        ]
        .mean()
    )

    return aggregate.round(
        4
    )


# ============================================================
# AFFICHAGE
# ============================================================

def print_results(
    df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    metric_columns: List[str],
    mode: str,
) -> None:

    print()
    print("=" * 80)
    print(
        "EVALUATION RAGAS - SYSTEME NBA"
    )
    print(
        f"MODE : {mode.upper()}"
    )
    print("=" * 80)

    print()

    print(
        f"Questions évaluées : {len(df)}"
    )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    if "pipeline_type" in df.columns:

        distribution = (
            df[
                "pipeline_type"
            ]
            .value_counts()
            .to_dict()
        )

        print(
            f"Répartition pipelines : "
            f"{distribution}"
        )

    # ========================================================
    # SCORES GLOBAUX
    # ========================================================

    print()
    print("-" * 80)
    print(
        "SCORES MOYENS"
    )
    print("-" * 80)

    for metric in metric_columns:

        values = pd.to_numeric(
            df[
                metric
            ],
            errors="coerce",
        ).dropna()

        if len(values) == 0:

            print(
                f"{metric:30s} : N/A"
            )

        else:

            print(
                f"{metric:30s} : "
                f"{values.mean():.4f}"
            )

    # ========================================================
    # PAR PIPELINE
    # ========================================================

    if (
        "pipeline_type" in df.columns
    ):

        print()
        print("-" * 80)
        print(
            "SCORES PAR PIPELINE"
        )
        print("-" * 80)

        pipeline_scores = (
            df.groupby(
                "pipeline_type"
            )[metric_columns]
            .mean()
            .round(4)
        )

        print(
            pipeline_scores.to_string(
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )

    # ========================================================
    # PAR CATEGORIE
    # ========================================================

    if not aggregate_df.empty:

        print()
        print("-" * 80)
        print(
            "SCORES PAR CATEGORIE"
        )
        print("-" * 80)

        print(
            aggregate_df.to_string(
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )

    print()


# ============================================================
# SAUVEGARDE
# ============================================================

def save_results(
    df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    mode: str,
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        ""
        if mode == "hybrid"
        else f"_{mode}"
    )

    # ========================================================
    # CSV DETAILLE
    # ========================================================

    detailed_path = (
        RESULTS_DIR
        / f"detailed_scores{suffix}.csv"
    )

    df.to_csv(
        detailed_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # CSV AGREGES
    # ========================================================

    aggregate_path = (
        RESULTS_DIR
        / f"aggregate_scores{suffix}.csv"
    )

    aggregate_df.to_csv(
        aggregate_path,
        encoding="utf-8-sig",
    )

    # ========================================================
    # JSON
    # ========================================================

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "model": MODEL_NAME,
        "search_k": SEARCH_K,
        "num_questions": len(df),
        "pipeline_distribution": (
            df[
                "pipeline_type"
            ]
            .value_counts()
            .to_dict()
        ),
        "metrics": [
            column
            for column
            in [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]
            if column in df.columns
        ],
        "aggregate_scores": {},
    }

    for category in aggregate_df.index:

        summary[
            "aggregate_scores"
        ][category] = {}

        for column in aggregate_df.columns:

            value = (
                aggregate_df.loc[
                    category,
                    column,
                ]
            )

            if pd.isna(value):

                summary[
                    "aggregate_scores"
                ][category][column] = None

            else:

                summary[
                    "aggregate_scores"
                ][category][column] = round(
                    float(value),
                    4,
                )

    summary_path = (
        RESULTS_DIR
        / f"evaluation_summary{suffix}.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Résultats sauvegardés dans %s",
        RESULTS_DIR,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Évaluation RAGAS du système "
            "hybride NBA RAG + SQL"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "rag",
            "hybrid",
        ],
        default="hybrid",
        help=(
            "hybrid = Router SQL + RAG "
            "(défaut), rag = RAG uniquement"
        ),
    )

    args = parser.parse_args()

    start_time = time.time()

    # ========================================================
    # VERIFICATION API KEY
    # ========================================================

    if not MISTRAL_API_KEY:

        raise RuntimeError(
            "MISTRAL_API_KEY absente."
        )

    # ========================================================
    # LOGFIRE GLOBAL
    # ========================================================

    with logfire.span(
        "evaluation-ragas-complete",
        mode=args.mode,
    ):

        # ====================================================
        # 1. QUESTIONS
        # ====================================================

        questions = (
            load_test_questions()
        )

        # ====================================================
        # 2. VECTOR STORE
        # ====================================================

        logger.info(
            "Initialisation VectorStoreManager..."
        )

        vector_store = (
            VectorStoreManager()
        )

        if vector_store.index is None:

            raise RuntimeError(
                "Index FAISS non chargé. "
                "Exécute 'python indexer.py'."
            )

        logger.info(
            "✓ VectorStoreManager chargé."
        )

        # ====================================================
        # 3. CLIENT MISTRAL
        # ====================================================

        mistral_client = (
            Mistral(
                api_key=MISTRAL_API_KEY
            )
        )

        logger.info(
            "✓ Client Mistral initialisé."
        )

        # ====================================================
        # 4. EXECUTION HYBRIDE
        # ====================================================

        results = (
            run_all_questions(
                questions=questions,
                vector_store=vector_store,
                mistral_client=mistral_client,
                mode=args.mode,
            )
        )

        # ====================================================
        # 5. LLM RAGAS
        # ====================================================

        print()
        print("=" * 70)
        print(
            "INITIALISATION RAGAS"
        )
        print("=" * 70)

        ragas_llm = (
            create_ragas_llm()
        )

        # ====================================================
        # 6. EMBEDDINGS
        # ====================================================

        ragas_embeddings = (
            create_ragas_embeddings()
        )

        # ====================================================
        # 7. METRIQUES
        # ====================================================

        metrics = (
            create_metrics(
                evaluator_llm=ragas_llm,
                evaluator_embeddings=ragas_embeddings,
            )
        )

        # ====================================================
        # 8. EVALUATION
        # ====================================================

        print()
        print("=" * 70)
        print(
            "EVALUATION RAGAS"
        )
        print(
            "SQL + RAG"
        )
        print("=" * 70)

        with logfire.span(
            "evaluation-metriques-ragas"
        ):

            df_scores = (
                run_evaluation(
                    results=results,
                    metrics=metrics,
                    evaluator_llm=ragas_llm,
                    evaluator_embeddings=ragas_embeddings,
                )
            )

        # ====================================================
        # 9. METADATA
        # ====================================================

        df_scores = (
            add_metadata(
                df_scores,
                results,
            )
        )

        # ====================================================
        # 10. METRIQUES
        # ====================================================

        metric_columns = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

        metric_columns = [
            column
            for column
            in metric_columns
            if column in df_scores.columns
        ]

        # ====================================================
        # 11. AGREGATS
        # ====================================================

        aggregate_df = (
            compute_aggregate_scores(
                df_scores,
                metric_columns,
            )
        )

        # ====================================================
        # 12. AFFICHAGE
        # ====================================================

        print_results(
            df=df_scores,
            aggregate_df=aggregate_df,
            metric_columns=metric_columns,
            mode=args.mode,
        )

        # ====================================================
        # 13. SAUVEGARDE
        # ====================================================

        save_results(
            df=df_scores,
            aggregate_df=aggregate_df,
            mode=args.mode,
        )

    # ========================================================
    # FIN
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 70)
    print(
        "EVALUATION TERMINEE"
    )
    print("=" * 70)

    print(
        f"Durée : {elapsed:.1f} secondes"
    )

    print(
        f"Questions évaluées : "
        f"{len(df_scores)}"
    )

    if "pipeline_type" in df_scores.columns:

        print(
            "SQL :",
            (
                df_scores[
                    "pipeline_type"
                ]
                == "SQL"
            ).sum(),
        )

        print(
            "RAG :",
            (
                df_scores[
                    "pipeline_type"
                ]
                == "RAG"
            ).sum(),
        )

    print()
    print(
        f"Résultats : {RESULTS_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()