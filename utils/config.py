# utils/config.py

import os
import logfire
from dotenv import load_dotenv


load_dotenv()


# ============================================================
# LOGFIRE
# ============================================================

LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

if LOGFIRE_TOKEN:
    logfire.configure(
        token=LOGFIRE_TOKEN,
    )


# ============================================================
# MISTRAL
# ============================================================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY n'est pas définie dans le fichier .env"
    )

MISTRAL_BASE_URL = os.getenv(
    "MISTRAL_BASE_URL",
    "https://api.mistral.ai/v1",
)

MODEL_NAME = os.getenv(
    "MODEL_ID",
    "mistral-small-latest",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "mistral-embed",
)


# ============================================================
# POSTGRESQL
# ============================================================

POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST",
    "localhost",
)

POSTGRES_PORT = os.getenv(
    "POSTGRES_PORT",
    "5434",
)

POSTGRES_DB = os.getenv(
    "POSTGRES_DB",
    "nba_rag",
)

POSTGRES_USER = os.getenv(
    "POSTGRES_USER",
    "postgres",
)

POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD"
)


if not POSTGRES_PASSWORD:
    raise ValueError(
        "POSTGRES_PASSWORD n'est pas définie dans le fichier .env"
    )


DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:"
    f"{POSTGRES_PASSWORD}@"
    f"{POSTGRES_HOST}:"
    f"{POSTGRES_PORT}/"
    f"{POSTGRES_DB}"
)


# ============================================================
# FICHIERS / VECTOR DB
# ============================================================

INPUT_DIR = os.getenv(
    "INPUT_DIR",
    "inputs",
)

VECTOR_DB_DIR = os.getenv(
    "OUTPUT_DIR",
    "vector_db",
)

FAISS_INDEX_FILE = os.getenv(
    "FAISS_INDEX_FILE",
    "vector_db/faiss_index.idx",
)

DOCUMENT_CHUNKS_FILE = os.getenv(
    "DOCUMENT_CHUNKS_FILE",
    "vector_db/document_chunks.pkl",
)


# ============================================================
# CONFIGURATION DES CHUNKS
# ============================================================

# Taille de chaque chunk en caractères
CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1500",
    )
)

# Nombre de caractères qui se chevauchent entre deux chunks
CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "150",
    )
)


# ============================================================
# EMBEDDINGS
# ============================================================

# Nombre de textes envoyés par lot à l'API d'embeddings
EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "EMBEDDING_BATCH_SIZE",
        "32",
    )
)


# ============================================================
# RECHERCHE
# ============================================================

SEARCH_K = int(
    os.getenv(
        "SEARCH_K",
        "5",
    )
)


# ============================================================
# APPLICATION
# ============================================================

APP_TITLE = "NBA Analyst AI"

NAME = "NBA"
