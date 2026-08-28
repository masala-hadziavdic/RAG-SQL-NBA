"""
Tool SQL LangChain pour interroger la base PostgreSQL NBA.

Architecture :
    Question utilisateur
        ↓
    Génération SQL par Mistral
        ↓
    Exécution PostgreSQL
        ↓
    Synthèse de la réponse

Usage :
    from database.sql_tool import query_nba_database

    result = query_nba_database(
        "Quel joueur a marqué le plus de points par match ?"
    )
"""

import logging
import os
import re
import sys

import logfire

from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI


# ============================================================
# IMPORT CONFIGURATION
# ============================================================

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

from utils.config import (
    DATABASE_URL,
    MISTRAL_API_KEY,
    MODEL_NAME,
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
# BASE DE DONNEES
# ============================================================

logger.info("Initialisation de la connexion SQL...")

db = SQLDatabase.from_uri(
    DATABASE_URL,
)

logger.info("Base PostgreSQL initialisée.")


# ============================================================
# LLM
# ============================================================

llm = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model=MODEL_NAME,
    temperature=0,
)


# ============================================================
# FEW-SHOT EXAMPLES
# ============================================================

FEW_SHOT_EXAMPLES = """
Exemples de questions et requêtes SQL :

Question :
Quel joueur a marqué le plus de points par match ?

SQL :
SELECT
    name,
    team_code,
    points_per_game
FROM players
WHERE points_per_game IS NOT NULL
ORDER BY points_per_game DESC
LIMIT 1;


Question :
Quels sont les 5 meilleurs marqueurs ?

SQL :
SELECT
    name,
    team_code,
    points_per_game
FROM players
WHERE points_per_game IS NOT NULL
ORDER BY points_per_game DESC
LIMIT 5;


Question :
Quel joueur a le plus de rebonds ?

SQL :
SELECT
    name,
    team_code,
    total_rebounds
FROM players
WHERE total_rebounds IS NOT NULL
ORDER BY total_rebounds DESC
LIMIT 1;


Question :
Quel joueur a le plus de passes décisives ?

SQL :
SELECT
    name,
    team_code,
    assists
FROM players
WHERE assists IS NOT NULL
ORDER BY assists DESC
LIMIT 1;


Question :
Quel est le meilleur marqueur des Lakers ?

SQL :
SELECT
    name,
    team_code,
    points_per_game
FROM players
WHERE team_code = 'LAL'
  AND points_per_game IS NOT NULL
ORDER BY points_per_game DESC
LIMIT 1;


Question :
Quels joueurs ont marqué plus de 25 points par match ?

SQL :
SELECT
    name,
    team_code,
    points_per_game
FROM players
WHERE points_per_game > 25
ORDER BY points_per_game DESC;


Question :
Compare Nikola Jokic et Giannis Antetokounmpo.

SQL :
SELECT
    name,
    team_code,
    points_per_game,
    total_rebounds,
    assists,
    field_goal_pct,
    true_shooting_pct
FROM players
WHERE name ILIKE '%Jokic%'
   OR name ILIKE '%Giannis%';


Question : Quelles sont les statistiques de Nikola Jokic ?
SQL : SELECT 
          name, 
          team_code, 
          games_played, 
          points_per_game, 
          total_rebounds, 
          assists, 
          field_goal_pct, 
          three_point_pct, 
          true_shooting_pct 
FROM players 
WHERE unaccent(name) 
ILIKE unaccent('%Nikola Jokic%');

Question : Quelles sont les statistiques de Luka Doncic ?
SQL : SELECT 
          name, 
          team_code, 
          games_played, 
          points_per_game, 
          total_rebounds, 
          assists, 
          field_goal_pct, 
          three_point_pct, 
          true_shooting_pct 
FROM players 
WHERE unaccent(name) 
ILIKE unaccent('%Luka Doncic%');

Question : Compare Nikola Jokic et Luka Doncic
SQL : SELECT 
          name, 
          team_code, 
          points_per_game, 
          total_rebounds, 
          assists, 
          field_goal_pct, 
          three_point_pct, 
          true_shooting_pct 
FROM players 
WHERE unaccent(name) 
ILIKE unaccent('%Nikola Jokic%') 
OR unaccent(name) 
ILIKE unaccent('%Luka Doncic%');

Question :
Quel joueur a le meilleur pourcentage à trois points ?

SQL :
SELECT
    name,
    team_code,
    three_point_pct
FROM players
WHERE three_point_pct IS NOT NULL
ORDER BY three_point_pct DESC
LIMIT 1;


Question :
Quel joueur a le meilleur true shooting percentage ?

SQL :
SELECT
    name,
    team_code,
    true_shooting_pct
FROM players
WHERE true_shooting_pct IS NOT NULL
  AND true_shooting_pct <= 100
ORDER BY true_shooting_pct DESC
LIMIT 1;


Question :
Combien de joueurs jouent pour les Lakers ?

SQL :
SELECT
    COUNT(*) AS nombre_joueurs
FROM players
WHERE team_code = 'LAL';


Question :
Quelles sont les équipes présentes dans la base ?

SQL :
SELECT
    code,
    name
FROM teams
ORDER BY name;


Question :
Quelle est la moyenne de points par équipe ?

SQL :
SELECT
    t.name AS team,
    AVG(p.points_per_game) AS moyenne_points
FROM players p
JOIN teams t
    ON p.team_code = t.code
WHERE p.points_per_game IS NOT NULL
GROUP BY t.code, t.name
ORDER BY moyenne_points DESC;


Question :
Quelle équipe a le meilleur marqueur ?

SQL :
SELECT
    t.name AS team,
    MAX(p.points_per_game) AS meilleur_scoreur
FROM players p
JOIN teams t
    ON p.team_code = t.code
WHERE p.points_per_game IS NOT NULL
GROUP BY t.code, t.name
ORDER BY meilleur_scoreur DESC;


Question :
Quels joueurs ont joué au moins 70 matchs et marqué plus de 20 points par match ?

SQL :
SELECT
    name,
    team_code,
    games_played,
    points_per_game
FROM players
WHERE games_played >= 70
  AND points_per_game > 20
ORDER BY points_per_game DESC;


Question :
Compare les meilleurs marqueurs et les meilleurs passeurs.

SQL :
SELECT
    name,
    team_code,
    points_per_game,
    assists
FROM players
WHERE points_per_game IS NOT NULL
   OR assists IS NOT NULL
ORDER BY points_per_game DESC
LIMIT 10;
"""


# ============================================================
# PROMPT GENERATION SQL
# ============================================================

SQL_GENERATION_PROMPT = PromptTemplate.from_template("""
Tu es un expert SQL PostgreSQL spécialisé dans l'analyse de statistiques NBA.

Ta tâche est de convertir une question en langage naturel en UNE requête SQL PostgreSQL valide.

SCHÉMA DE LA BASE DE DONNÉES :
{schema}

EXEMPLES DE QUESTIONS ET REQUÊTES SQL :
{few_shot_examples}

RÈGLES IMPORTANTES :

1. Génère UNIQUEMENT la requête SQL.
   Ne génère aucune explication, aucun commentaire et aucun bloc Markdown.

2. Utilise uniquement les tables et colonnes présentes dans le schéma.

3. La base utilise PostgreSQL.

4. Pour rechercher un joueur par son nom, utilise TOUJOURS
   la fonction PostgreSQL unaccent avec ILIKE :

   unaccent(name) ILIKE unaccent('%nom recherché%')

   Cette règle permet de rechercher les joueurs sans tenir compte des accents.

   Exemple :
   unaccent(name) ILIKE unaccent('%Nikola Jokic%')
   doit permettre de trouver "Nikola Jokić".

   Exemple :
   unaccent(name) ILIKE unaccent('%Luka Doncic%')
   doit permettre de trouver "Luka Dončić".

5. Ne recherche jamais un joueur uniquement avec une égalité exacte
   lorsque la question contient son nom.

6. Les codes d'équipes sont généralement des codes NBA à trois lettres
   comme LAL, BOS, OKC, DEN, MIL, etc.

7. Pour une question demandant le meilleur ou le pire joueur,
   utilise ORDER BY avec LIMIT 1 lorsque cela est approprié.

8. Pour un classement des meilleurs joueurs, utilise ORDER BY et LIMIT.

9. Pour les agrégations, utilise les fonctions SQL appropriées :
   COUNT, SUM, AVG, MIN, MAX, etc.

10. Pour comparer plusieurs joueurs, retourne les lignes correspondant
    aux joueurs demandés.

11. N'invente jamais une colonne.
    Par exemple, utilise points_per_game pour les points par match
    et total_points pour le total des points.

12. Pour les pourcentages, conserve les valeurs entre 0 et 100.

13. Pour les statistiques de tir, fais attention aux petits échantillons.
    Un joueur ayant 100 % avec une seule tentative ne doit pas
    automatiquement être considéré comme le meilleur tireur si la question
    implique implicitement la performance sur un volume significatif.

14. Si la question demande explicitement une statistique par match,
utilise la colonne correspondante *_per_game lorsqu'elle existe.

Si cette colonne n'existe pas, calcule la moyenne par match
à partir de la valeur totale divisée par games_played.

15. DISTINCTION ENTRE STATISTIQUES TOTALES ET STATISTIQUES PAR MATCH :

- total_points = nombre TOTAL de points sur la saison.
- total_rebounds = nombre TOTAL de rebonds sur la saison.
- assists = nombre TOTAL de passes décisives sur la saison.
- games_played = nombre TOTAL de matchs joués.
- points_per_game = moyenne de points par match.

Lorsque la question demande une statistique "par match",
ne jamais présenter total_points, total_rebounds ou assists
comme une statistique par match.

Si une colonne *_per_game n'existe pas pour une statistique,
calcule-la à partir de la statistique totale divisée par games_played.

Par exemple :

ROUND((total_rebounds / NULLIF(games_played, 0))::numeric, 2)
AS rebounds_per_game

ROUND((assists / NULLIF(games_played, 0))::numeric, 2)
AS assists_per_game

16. Lorsque plusieurs statistiques sont retournées,
conserve toujours les colonnes dans un ordre explicite et cohérent
et utilise des alias explicites pour éviter toute confusion.

Exemple :
total_rebounds AS total_rebounds,
ROUND((total_rebounds / NULLIF(games_played, 0))::numeric, 2)
AS rebounds_per_game,
assists AS total_assists,
ROUND((assists / NULLIF(games_played, 0))::numeric, 2)
AS assists_per_game.

17. Ne jamais appeler une statistique "par match" si elle représente
une valeur totale.

18. Si la question demande les statistiques générales d'un joueur,
tu peux retourner à la fois les valeurs totales et les moyennes par match,
mais leurs noms doivent permettre de les distinguer clairement.

19. Utilise LIMIT lorsque la question demande un nombre précis de résultats.

20. Génère uniquement des requêtes de lecture SELECT.
    Ne génère jamais :
    INSERT, UPDATE, DELETE, DROP, ALTER, CREATE ou TRUNCATE.

QUESTION UTILISATEUR :
{question}

SQL :
""")


# ============================================================
# PROMPT SYNTHESE
# ============================================================

ANSWER_SYNTHESIS_PROMPT = PromptTemplate.from_template(
    """
Tu es un analyste NBA chargé de formuler une réponse à partir d'un résultat PostgreSQL.

Tu dois répondre à la question de l'utilisateur UNIQUEMENT à partir du résultat SQL fourni.

========================
QUESTION UTILISATEUR
========================
{question}

========================
RÉSULTAT SQL
========================
{sql_result}

========================
RÈGLES ABSOLUES
========================

1. SOURCE UNIQUE
- Utilise uniquement les données présentes dans {sql_result}.
- N'utilise aucune connaissance externe.
- N'invente aucune statistique.
- Ne complète jamais une donnée absente.
- Ne fais aucune supposition.
- Si une information n'est pas présente dans le résultat SQL, ne la mentionne pas.

2. RESPECT STRICT DES COLONNES
- Chaque valeur doit être associée UNIQUEMENT au nom de la colonne SQL correspondant à cette valeur.
- Ne déduis JAMAIS la signification d'une valeur à partir de sa position dans la liste.
- Ne change JAMAIS le nom ou le sens d'une colonne.
- Ne permute jamais les valeurs entre les colonnes.
- Si plusieurs colonnes numériques sont présentes, vérifie attentivement leur nom avant de les présenter.

Exemple :

SQL :
SELECT name, team_code, games_played, points_per_game,
       total_rebounds, assists, field_goal_pct
FROM players;

Résultat :
[('Nikola Jokić', 'DEN', 70, 29.6, 889, 714, 57.6)]

La réponse correcte est :

- Matchs joués : 70
- Points par match : 29,6
- Rebonds totaux : 889
- Passes décisives totales : 714
- Pourcentage de réussite aux tirs : 57,6 %

Il est INTERDIT de transformer :
- 889 en passes décisives
- 714 en rebonds
- 57,6 en nombre de matchs

3. SENS DES COLONNES PLAYERS

Voici le sens exact des colonnes :

- name = nom du joueur
- team_code = code de l'équipe
- age = âge
- games_played = nombre de matchs joués
- wins = nombre de victoires
- losses = nombre de défaites
- minutes_per_game = minutes par match
- points_per_game = moyenne de points par match
- field_goals_made = tirs réussis
- field_goals_attempted = tirs tentés
- field_goal_pct = pourcentage de réussite aux tirs
- three_pointers_made = tirs à trois points réussis
- three_pointers_attempted = tirs à trois points tentés
- three_point_pct = pourcentage de réussite à trois points
- free_throws_made = lancers francs réussis
- free_throws_attempted = lancers francs tentés
- free_throw_pct = pourcentage de réussite aux lancers francs
- offensive_rebounds = rebonds offensifs
- defensive_rebounds = rebonds défensifs
- total_rebounds = total de rebonds
- assists = total de passes décisives
- turnovers = total de pertes de balle
- steals = total d'interceptions
- blocks = total de contres
- personal_fouls = total de fautes personnelles
- fantasy_points = points fantasy
- double_doubles = nombre de double-doubles
- triple_doubles = nombre de triple-doubles
- plus_minus = +/- 
- offensive_rating = rating offensif
- defensive_rating = rating défensif
- net_rating = net rating
- assist_pct = pourcentage de passes décisives
- assist_to_turnover = ratio passes/pertes de balle
- assist_ratio = ratio de passes
- offensive_rebound_pct = pourcentage de rebonds offensifs
- defensive_rebound_pct = pourcentage de rebonds défensifs
- total_rebound_pct = pourcentage de rebonds totaux
- turnover_ratio = ratio de pertes de balle
- effective_fg_pct = pourcentage de réussite effective
- true_shooting_pct = pourcentage de true shooting
- usage_rate = taux d'utilisation
- pace = rythme
- player_impact_estimate = Player Impact Estimate
- possessions = possessions
- total_points = total de points sur la saison

4. TOTAL VS MOYENNE

Respecte strictement cette distinction :

- points_per_game = points PAR MATCH
- total_points = points AU TOTAL / SUR LA SAISON
- total_rebounds = rebonds AU TOTAL / SUR LA SAISON
- assists = passes décisives AU TOTAL / SUR LA SAISON
- games_played = nombre de matchs joués

NE FAIS JAMAIS de conversion entre ces valeurs.

Par exemple :

Si SQL retourne :
points_per_game = 32.69736842105263

Réponds :
"32,7 points par match"

et JAMAIS :
"32 697 points"
ou
"32,7 points au total".

Si SQL retourne :
total_points = 2485

Réponds :
"2 485 points au total"
ou
"2 485 points sur la saison".

Si SQL retourne :
assists = 714

Réponds :
"714 passes décisives au total"
ou
"714 passes décisives sur la saison".

JAMAIS :
"714 passes décisives par match".

5. AGRÉGATIONS SQL

Attention aux fonctions SQL comme :

MAX(points_per_game)
AVG(points_per_game)
SUM(total_points)
MAX(three_point_pct)

Le résultat doit conserver le sens de la colonne utilisée.

Exemple :

SQL :
MAX(p.points_per_game) AS meilleur_scoreur

Résultat :
[('Oklahoma City Thunder', 32.69736842105263)]

La réponse correcte est :

"L'Oklahoma City Thunder a le meilleur marqueur avec 32,7 points par match."

Il est INTERDIT de répondre :

"L'Oklahoma City Thunder a marqué 32 697 points."

La valeur vient de points_per_game et signifie donc une moyenne de points par match.

6. POURCENTAGES

Pour les colonnes suivantes, ajoute le symbole % :

- field_goal_pct
- three_point_pct
- free_throw_pct
- true_shooting_pct
- assist_pct
- offensive_rebound_pct
- defensive_rebound_pct
- total_rebound_pct
- turnover_ratio si le résultat est explicitement présenté comme un pourcentage
- effective_fg_pct
- usage_rate

Ne transforme pas une valeur décimale en pourcentage supplémentaire.

Exemple :

SQL :
three_point_pct = 41.7

Réponse :
"41,7 %"

JAMAIS :
"4 170 %".

7. ARRONDI

Pour les valeurs décimales :

- Arrondis les moyennes et pourcentages à 1 décimale lorsque cela améliore la lisibilité.
- Exemple : 32.69736842105263 → 32,7
- Exemple : 30.402985074626866 → 30,4
- Exemple : 100.0 → 100,0 %

Pour les valeurs entières :

- Ne rajoute pas de décimales inutiles.
- Exemple : 70 → 70 matchs
- Exemple : 889.0 → 889 rebonds

8. STATISTIQUES D'UN JOUEUR

Si plusieurs statistiques sont retournées, présente uniquement les colonnes effectivement présentes dans le résultat SQL.

Exemple :

Résultat :
[
    ('Nikola Jokić', 'DEN', 70, 29.6, 889.0, 714.0,
     57.6, 41.7, 80.0, 66.3, 28.5, 20.6)
]

Si les colonnes SQL sont :

name,
team_code,
games_played,
points_per_game,
total_rebounds,
assists,
field_goal_pct,
three_point_pct,
free_throw_pct,
true_shooting_pct,
usage_rate,
player_impact_estimate

Alors la réponse doit respecter EXACTEMENT cette correspondance.

Il ne faut notamment PAS inventer :
- total_points
- rebounds_per_game
- assists_per_game
- une deuxième valeur de points_per_game
- une autre statistique absente du SQL.

9. COMPARAISONS

Pour une comparaison entre plusieurs joueurs :

- Compare uniquement les statistiques présentes dans le résultat SQL.
- Conserve le sens exact de chaque colonne.
- Pour chaque joueur, associe chaque valeur à la bonne colonne.
- Ne mélange jamais les valeurs entre joueurs.

Exemple :

Nikola Jokić :
points_per_game = 29.6
total_rebounds = 889
assists = 714
field_goal_pct = 57.6

Giannis Antetokounmpo :
points_per_game = 30.4
total_rebounds = 797
assists = 436
field_goal_pct = 60.1

La réponse doit conserver exactement ces correspondances.

10. QUESTIONS "MEILLEUR"

Si le SQL retourne une seule ligne parce qu'il utilise ORDER BY ... DESC LIMIT 1 :

- Identifie le joueur ou l'équipe à partir de la colonne correspondante.
- Présente la valeur avec le sens exact de la colonne.

Exemple :

SQL :
name, team_code, three_point_pct

Résultat :
Alondes Williams, DET, 100.0

Réponse :
"Le joueur avec le meilleur pourcentage à trois points est Alondes Williams (DET) avec 100,0 %."

11. ÉQUIPE

Si le résultat contient :

team
meilleur_scoreur

et que meilleur_scoreur provient de :

MAX(points_per_game)

alors "meilleur_scoreur" signifie :

"meilleur marqueur en moyenne de points par match"

et NON :

"total de points de l'équipe".

12. DONNÉES ABSENTES

Si {sql_result} est :
[]
ou ne contient aucune ligne exploitable,

réponds simplement :

"Aucune donnée correspondante n'a été trouvée."

Ne tente pas de répondre à partir de connaissances générales.

13. FORMAT DE RÉPONSE

- Réponds en français.
- Sois clair, précis et concis.
- Utilise le nom réel du joueur ou de l'équipe présent dans le résultat SQL.
- Utilise le code de l'équipe s'il est fourni.
- Utilise des listes à puces pour plusieurs statistiques.
- N'ajoute aucune explication inutile.
- Ne mentionne jamais que tu es un modèle de langage.
- Ne mentionne jamais des données qui ne sont pas présentes dans le résultat SQL.

========================
RÈGLE FINALE
========================

AVANT DE RÉPONDRE, vérifie mentalement chaque valeur :

1. Quel est le nom EXACT de la colonne SQL ?
2. Quelle est la signification EXACTE de cette colonne ?
3. Est-ce une moyenne, un total, un nombre de matchs ou un pourcentage ?
4. Est-ce que je suis en train d'inventer ou de supposer une information ?
5. Est-ce que chaque valeur est associée au BON nom de colonne ?

Si une réponse ne respecte pas exactement le résultat SQL, corrige-la avant de l'envoyer.
"""
)

# ============================================================
# SCHEMA
# ============================================================

def get_schema() -> str:
    """
    Retourne le schéma des tables utilisées par le Tool SQL.
    """

    return db.get_table_info(
        table_names=[
            "players",
            "teams",
        ]
    )


# ============================================================
# NETTOYAGE SQL
# ============================================================

def clean_sql_query(sql_query: str) -> str:
    """
    Nettoie la réponse du LLM pour récupérer uniquement
    la requête SQL.
    """

    sql_query = sql_query.strip()

    # Retirer les blocs Markdown éventuels
    sql_query = re.sub(
        r"^```sql\s*",
        "",
        sql_query,
        flags=re.IGNORECASE,
    )

    sql_query = re.sub(
        r"^```\s*",
        "",
        sql_query,
    )

    sql_query = re.sub(
        r"\s*```$",
        "",
        sql_query,
    )

    return sql_query.strip()


# ============================================================
# VALIDATION SECURITE SQL
# ============================================================

def validate_sql_query(sql_query: str) -> None:
    """
    Vérifie que la requête générée est une requête SELECT.

    Le Tool SQL ne doit jamais modifier la base.
    """

    normalized = sql_query.strip().lower()

    if not normalized.startswith("select"):
        raise ValueError(
            "Sécurité SQL : seule une requête SELECT est autorisée."
        )

    forbidden_keywords = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "grant",
        "revoke",
    ]

    for keyword in forbidden_keywords:

        if re.search(
            rf"\b{keyword}\b",
            normalized,
        ):
            raise ValueError(
                f"Requête SQL interdite : {keyword.upper()}"
            )


# ============================================================
# GENERATION SQL
# ============================================================

def generate_sql_query(
    question: str,
) -> str:
    """
    Génère une requête SQL PostgreSQL à partir
    d'une question en langage naturel.
    """

    with logfire.span(
        "generation-sql",
        question=question[:100],
    ):

        schema = get_schema()

        prompt = SQL_GENERATION_PROMPT.format(
            schema=schema,
            few_shot_examples=FEW_SHOT_EXAMPLES,
            question=question,
        )

        response = llm.invoke(prompt)

        sql_query = clean_sql_query(
            response.content
        )

        validate_sql_query(
            sql_query
        )

        logger.info(
            "SQL généré : %s",
            sql_query,
        )

        logfire.info(
            "Requête SQL générée",
            sql=sql_query,
        )

        return sql_query


# ============================================================
# EXECUTION SQL
# ============================================================

def execute_sql_query(
    sql_query: str,
) -> str:
    """
    Exécute la requête SQL PostgreSQL.
    """

    with logfire.span(
        "execution-sql",
        sql=sql_query[:100],
    ):

        try:

            result = db.run(
                sql_query
            )

            logger.info(
                "Requête SQL exécutée avec succès."
            )

            return str(result)

        except Exception as exc:

            error_msg = (
                f"Erreur SQL : {exc}"
            )

            logger.error(
                error_msg
            )

            logfire.error(
                "Erreur exécution SQL",
                erreur=str(exc),
            )

            return error_msg


# ============================================================
# SYNTHESE REPONSE
# ============================================================

def synthesize_answer(
    question: str,
    sql_result: str,
) -> str:
    """
    Transforme le résultat SQL en réponse naturelle.
    """

    with logfire.span(
        "synthese-reponse",
    ):

        prompt = ANSWER_SYNTHESIS_PROMPT.format(
            question=question,
            sql_result=sql_result,
        )

        response = llm.invoke(
            prompt
        )

        answer = response.content.strip()

        logger.info(
            "Réponse SQL générée."
        )

        return answer


# ============================================================
# TOOL PRINCIPAL
# ============================================================

def query_nba_database(
    question: str,
) -> dict:
    """
    Tool SQL principal.

    Pipeline :

        Question
            ↓
        Génération SQL
            ↓
        Validation
            ↓
        PostgreSQL
            ↓
        Synthèse LLM

    Returns:
        {
            "question": ...,
            "sql_query": ...,
            "sql_result": ...,
            "answer": ...
        }
    """

    with logfire.span(
        "tool-sql-nba",
        question=question[:100],
    ):

        logger.info(
            "Question SQL reçue : %s",
            question,
        )

        # 1. Génération SQL
        sql_query = generate_sql_query(
            question
        )

        # 2. Exécution
        sql_result = execute_sql_query(
            sql_query
        )

        # 3. Synthèse
        
        answer = synthesize_answer(
            question,
            sql_result,
        )

    return {
        "question": question,
        "sql_query": sql_query,
        "sql_result": sql_result,
        "answer": answer,
    }

# ============================================================
# TEST LOCAL
# ============================================================

if __name__ == "__main__":

    test_questions = [

        "Quel joueur a marqué le plus de points par match ?",

        "Quels sont les 5 meilleurs marqueurs ?",

        "Quel est le meilleur marqueur des Lakers ?",

        "Quelles sont les statistiques de Nikola Jokic ?",

        "Compare Nikola Jokic et Giannis Antetokounmpo.",

        "Quel joueur a le meilleur pourcentage à trois points ?",

        "Quelle équipe a le meilleur marqueur ?",

    ]

    for question in test_questions:

        print()
        print("=" * 70)
        print(f"QUESTION : {question}")
        print("=" * 70)

        try:

            result = query_nba_database(
                question
            )

            print()
            print("SQL :")
            print(result["sql_query"])

            print()
            print("RESULTAT SQL :")
            print(result["sql_result"])

            print()
            print("REPONSE :")
            print(result["answer"])

        except Exception as exc:

            print()
            print("ERREUR :")
            print(exc)
            
 # ============================================================
# TEST DU SCHEMA
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("TABLES DISPONIBLES")
    print("=" * 70)

    print(db.get_usable_table_names())

    print("\n" + "=" * 70)
    print("SCHEMA DE LA BASE")
    print("=" * 70)

    print(db.get_table_info())