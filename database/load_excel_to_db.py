"""
load_excel_to_db.py

Import des données NBA depuis Excel vers PostgreSQL.

Source :
    inputs/regular NBA.xlsx

Base :
    PostgreSQL / nba_rag

Tables alimentées :
    - teams
    - players

Les tables matches, stats et reports ne sont pas alimentées
car le fichier Excel fourni contient des statistiques de saison
et non des données match par match.
"""

from pathlib import Path
import logging
import os
import re
from datetime import time
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EXCEL_FILE = BASE_DIR / "inputs" / "regular NBA.xlsx"

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5434")),
    "dbname": os.getenv("POSTGRES_DB", "nba_rag"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}

SHEET_NAME = "Données NBA"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# MODELE PYDANTIC
# ============================================================

class PlayerData(BaseModel):
    """
    Modèle de validation d'un joueur avant insertion PostgreSQL.
    """

    id: int
    name: str
    team_code: Optional[str] = None

    age: Optional[int] = Field(default=None, ge=0)

    games_played: Optional[int] = Field(default=None, ge=0)
    wins: Optional[int] = Field(default=None, ge=0)
    losses: Optional[int] = Field(default=None, ge=0)

    minutes_per_game: Optional[float] = None

    total_points: Optional[float] = None
    points_per_game: Optional[float] = None

    field_goals_made: Optional[float] = None
    field_goals_attempted: Optional[float] = None
    field_goal_pct: Optional[float] = None

    three_pointers_made: Optional[float] = None
    three_pointers_attempted: Optional[float] = None
    three_point_pct: Optional[float] = None

    free_throws_made: Optional[float] = None
    free_throws_attempted: Optional[float] = None
    free_throw_pct: Optional[float] = None

    offensive_rebounds: Optional[float] = None
    defensive_rebounds: Optional[float] = None
    total_rebounds: Optional[float] = None

    assists: Optional[float] = None
    turnovers: Optional[float] = None

    steals: Optional[float] = None
    blocks: Optional[float] = None

    personal_fouls: Optional[float] = None

    fantasy_points: Optional[float] = None

    double_doubles: Optional[int] = None
    triple_doubles: Optional[int] = None

    plus_minus: Optional[float] = None

    offensive_rating: Optional[float] = None
    defensive_rating: Optional[float] = None
    net_rating: Optional[float] = None

    assist_pct: Optional[float] = None
    assist_to_turnover: Optional[float] = None
    assist_ratio: Optional[float] = None

    offensive_rebound_pct: Optional[float] = None
    defensive_rebound_pct: Optional[float] = None
    total_rebound_pct: Optional[float] = None

    turnover_ratio: Optional[float] = None

    effective_fg_pct: Optional[float] = None
    true_shooting_pct: Optional[float] = None

    usage_rate: Optional[float] = None
    pace: Optional[float] = None

    player_impact_estimate: Optional[float] = None

    possessions: Optional[float] = None


# ============================================================
# UTILITAIRES
# ============================================================

def clean_value(value):
    """
    Nettoie une valeur Excel.

    NaN -> None
    time(15:00) -> 15.0
    """
    if pd.isna(value):
        return None

    if isinstance(value, time):
        return float(value.hour)

    return value


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage général du DataFrame Excel.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Correction du nom de la colonne 3PM
    # --------------------------------------------------------

    corrected_columns = []

    for col in df.columns:

        if isinstance(col, time):
            # 15:00:00 correspond ici à 3PM
            if col.hour == 15 and col.minute == 0:
                corrected_columns.append("3PM")
            else:
                corrected_columns.append(str(col))
        else:
            corrected_columns.append(str(col).strip())

    df.columns = corrected_columns

    # --------------------------------------------------------
    # Supprimer les colonnes Unnamed complètement vides
    # --------------------------------------------------------

    unnamed_columns = [
        col
        for col in df.columns
        if col.startswith("Unnamed:")
    ]

    if unnamed_columns:
        logger.info(
            "Suppression de %d colonnes vides.",
            len(unnamed_columns),
        )

        df = df.drop(columns=unnamed_columns)

    # --------------------------------------------------------
    # Nettoyage des cellules
    # --------------------------------------------------------

    df = df.map(clean_value)

    # --------------------------------------------------------
    # Nettoyage des noms
    # --------------------------------------------------------

    df["Player"] = (
        df["Player"]
        .astype(str)
        .str.strip()
    )

    df["Team"] = (
        df["Team"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return df


# ============================================================
# MAPPING EXCEL -> POSTGRESQL
# ============================================================

COLUMN_MAPPING = {
    "Player": "name",
    "Team": "team_code",
    "Age": "age",

    "GP": "games_played",
    "W": "wins",
    "L": "losses",

    "Min": "minutes_per_game",

    "PTS": "total_points",

    "FGM": "field_goals_made",
    "FGA": "field_goals_attempted",
    "FG%": "field_goal_pct",

    "3PM": "three_pointers_made",
    "3PA": "three_pointers_attempted",
    "3P%": "three_point_pct",

    "FTM": "free_throws_made",
    "FTA": "free_throws_attempted",
    "FT%": "free_throw_pct",

    "OREB": "offensive_rebounds",
    "DREB": "defensive_rebounds",
    "REB": "total_rebounds",

    "AST": "assists",
    "TOV": "turnovers",

    "STL": "steals",
    "BLK": "blocks",

    "PF": "personal_fouls",

    "FP": "fantasy_points",

    "DD2": "double_doubles",
    "TD3": "triple_doubles",

    "+/-": "plus_minus",

    "OFFRTG": "offensive_rating",
    "DEFRTG": "defensive_rating",
    "NETRTG": "net_rating",

    "AST%": "assist_pct",
    "AST/TO": "assist_to_turnover",
    "AST RATIO": "assist_ratio",

    "OREB%": "offensive_rebound_pct",
    "DREB%": "defensive_rebound_pct",
    "REB%": "total_rebound_pct",

    "TO RATIO": "turnover_ratio",

    "EFG%": "effective_fg_pct",
    "TS%": "true_shooting_pct",

    "USG%": "usage_rate",
    "PACE": "pace",

    "PIE": "player_impact_estimate",

    "POSS": "possessions",
}


# ============================================================
# CHARGEMENT EXCEL
# ============================================================

def load_excel() -> pd.DataFrame:

    logger.info("Lecture du fichier Excel : %s", EXCEL_FILE)

    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            f"Fichier Excel introuvable : {EXCEL_FILE}"
        )

    df = pd.read_excel(
        EXCEL_FILE,
        sheet_name=SHEET_NAME,
        header=1,
    )

    logger.info(
        "Excel chargé : %d lignes × %d colonnes",
        df.shape[0],
        df.shape[1],
    )

    df = clean_dataframe(df)

    logger.info(
        "Après nettoyage : %d lignes × %d colonnes",
        df.shape[0],
        df.shape[1],
    )

    return df


# ============================================================
# PREPARATION PLAYERS
# ============================================================

def prepare_players(df: pd.DataFrame) -> list[PlayerData]:

    logger.info("Préparation des joueurs...")

    # Vérification des colonnes nécessaires
    required_columns = set(COLUMN_MAPPING.keys())

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes Excel manquantes : {sorted(missing)}"
        )

    # Renommage
    players_df = df.rename(
        columns=COLUMN_MAPPING
    ).copy()

    # --------------------------------------------------------
    # ID interne
    # --------------------------------------------------------

    players_df.insert(
        0,
        "id",
        range(1, len(players_df) + 1)
    )

    # --------------------------------------------------------
    # Conversion numérique
    # --------------------------------------------------------

    integer_columns = [
        "id",
        "age",
        "games_played",
        "wins",
        "losses",
        "double_doubles",
        "triple_doubles",
    ]

    for col in integer_columns:
        players_df[col] = pd.to_numeric(
            players_df[col],
            errors="coerce",
        )
    # --------------------------------------------------------
    # CALCUL DES POINTS PAR MATCH
    # --------------------------------------------------------

    players_df["points_per_game"] = (
        players_df["total_points"]
        / players_df["games_played"].replace(0, pd.NA)
    )

    # --------------------------------------------------------
    # Validation Pydantic
    # --------------------------------------------------------

    validated_players = []

    errors = []

    for index, row in players_df.iterrows():

        data = row.to_dict()

        try:
            player = PlayerData(**data)
            validated_players.append(player)

        except ValidationError as exc:
            errors.append(
                {
                    "row": index + 2,
                    "player": data.get("name"),
                    "error": str(exc),
                }
            )

    if errors:

        logger.warning(
            "%d lignes invalides détectées.",
            len(errors),
        )

        for error in errors[:10]:
            logger.warning(error)

    logger.info(
        "%d joueurs validés avec succès.",
        len(validated_players),
    )

    return validated_players


# ============================================================
# CONNEXION POSTGRESQL
# ============================================================

def get_connection():

    logger.info(
        "Connexion PostgreSQL : %s:%s/%s",
        DB_CONFIG["host"],
        DB_CONFIG["port"],
        DB_CONFIG["dbname"],
    )

    return psycopg2.connect(
        **DB_CONFIG
    )


# ============================================================
# INSERT TEAMS
# ============================================================

def insert_teams(
    conn,
    players: list[PlayerData],
):

    teams = {}

    for player in players:

        if player.team_code:

            teams[player.team_code] = player.team_code

    # Pour le moment, on ne connaît pas les noms complets
    # depuis la feuille Données NBA.
    #
    # Les noms sont disponibles dans la feuille "Equipe".
    #
    # Cette fonction sera donc alimentée par load_teams()
    # ci-dessous.

    return teams


# ============================================================
# CHARGEMENT DES EQUIPES
# ============================================================

def load_teams_from_excel():

    logger.info("Lecture de la feuille Equipe...")

    df = pd.read_excel(
        EXCEL_FILE,
        sheet_name="Equipe",
    )

    df.columns = [
        "code",
        "name",
    ]

    df["code"] = (
        df["code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["name"] = (
        df["name"]
        .astype(str)
        .str.strip()
    )

    teams = []

    for _, row in df.iterrows():

        if not row["code"]:
            continue

        teams.append(
            (
                row["code"],
                row["name"],
            )
        )

    logger.info(
        "%d équipes trouvées.",
        len(teams),
    )

    return teams


def insert_teams_into_db(
    conn,
    teams,
):

    query = """
        INSERT INTO teams (code, name)
        VALUES %s
        ON CONFLICT (code)
        DO UPDATE SET
            name = EXCLUDED.name
    """

    with conn.cursor() as cursor:

        execute_values(
            cursor,
            query,
            teams,
        )

    logger.info(
        "%d équipes insérées/mises à jour.",
        len(teams),
    )


# ============================================================
# INSERT PLAYERS
# ============================================================

def insert_players_into_db(
    conn,
    players: list[PlayerData],
):

    columns = [
        field
        for field in PlayerData.model_fields.keys()
    ]

    column_sql = ", ".join(columns)

    values = []

    for player in players:

        data = player.model_dump()

        values.append(
            tuple(
                data[column]
                for column in columns
            )
        )

    query = f"""
        INSERT INTO players (
            {column_sql}
        )
        VALUES %s

        ON CONFLICT (id)
        DO UPDATE SET

            name = EXCLUDED.name,
            team_code = EXCLUDED.team_code,
            age = EXCLUDED.age,

            games_played = EXCLUDED.games_played,
            wins = EXCLUDED.wins,
            losses = EXCLUDED.losses,

            minutes_per_game = EXCLUDED.minutes_per_game,

            total_points = EXCLUDED.total_points,
            points_per_game = EXCLUDED.points_per_game,

            field_goals_made = EXCLUDED.field_goals_made,
            field_goals_attempted = EXCLUDED.field_goals_attempted,
            field_goal_pct = EXCLUDED.field_goal_pct,

            three_pointers_made = EXCLUDED.three_pointers_made,
            three_pointers_attempted = EXCLUDED.three_pointers_attempted,
            three_point_pct = EXCLUDED.three_point_pct,

            free_throws_made = EXCLUDED.free_throws_made,
            free_throws_attempted = EXCLUDED.free_throws_attempted,
            free_throw_pct = EXCLUDED.free_throw_pct,

            offensive_rebounds = EXCLUDED.offensive_rebounds,
            defensive_rebounds = EXCLUDED.defensive_rebounds,
            total_rebounds = EXCLUDED.total_rebounds,

            assists = EXCLUDED.assists,
            turnovers = EXCLUDED.turnovers,

            steals = EXCLUDED.steals,
            blocks = EXCLUDED.blocks,

            personal_fouls = EXCLUDED.personal_fouls,

            fantasy_points = EXCLUDED.fantasy_points,

            double_doubles = EXCLUDED.double_doubles,
            triple_doubles = EXCLUDED.triple_doubles,

            plus_minus = EXCLUDED.plus_minus,

            offensive_rating = EXCLUDED.offensive_rating,
            defensive_rating = EXCLUDED.defensive_rating,
            net_rating = EXCLUDED.net_rating,

            assist_pct = EXCLUDED.assist_pct,
            assist_to_turnover = EXCLUDED.assist_to_turnover,
            assist_ratio = EXCLUDED.assist_ratio,

            offensive_rebound_pct = EXCLUDED.offensive_rebound_pct,
            defensive_rebound_pct = EXCLUDED.defensive_rebound_pct,
            total_rebound_pct = EXCLUDED.total_rebound_pct,

            turnover_ratio = EXCLUDED.turnover_ratio,

            effective_fg_pct = EXCLUDED.effective_fg_pct,
            true_shooting_pct = EXCLUDED.true_shooting_pct,

            usage_rate = EXCLUDED.usage_rate,
            pace = EXCLUDED.pace,

            player_impact_estimate = EXCLUDED.player_impact_estimate,

            possessions = EXCLUDED.possessions
    """

    with conn.cursor() as cursor:

        execute_values(
            cursor,
            query,
            values,
        )

    logger.info(
        "%d joueurs insérés/mis à jour.",
        len(players),
    )

# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("=" * 70)
    logger.info("IMPORT EXCEL → POSTGRESQL")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # 1. Charger Excel
    # --------------------------------------------------------

    df = load_excel()

    # --------------------------------------------------------
    # 2. Préparer + valider players
    # --------------------------------------------------------

    players = prepare_players(df)

    if not players:
        raise RuntimeError(
            "Aucun joueur valide à importer."
        )

    # --------------------------------------------------------
    # 3. Charger équipes
    # --------------------------------------------------------

    teams = load_teams_from_excel()

    # --------------------------------------------------------
    # 4. Connexion PostgreSQL
    # --------------------------------------------------------

    conn = None

    try:

        conn = get_connection()

        # ----------------------------------------------------
        # Transaction
        # ----------------------------------------------------

        insert_teams_into_db(
            conn,
            teams,
        )

        insert_players_into_db(
            conn,
            players,
        )

        conn.commit()

        logger.info("=" * 70)
        logger.info("IMPORT TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 70)

        logger.info(
            "Équipes : %d",
            len(teams),
        )

        logger.info(
            "Joueurs : %d",
            len(players),
        )

        logger.info(
            "Base : %s",
            DB_CONFIG["dbname"],
        )

    except Exception as exc:

        if conn:
            conn.rollback()

        logger.exception(
            "ERREUR IMPORT : %s",
            exc,
        )

        raise

    finally:

        if conn:
            conn.close()

            logger.info(
                "Connexion PostgreSQL fermée."
            )


if __name__ == "__main__":
    main()