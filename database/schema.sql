-- ============================================================
-- SCHEMA POSTGRESQL - RAG + SQL
-- Projet : NBA
-- Base : nba_rag
--
-- Architecture :
--
-- teams
--   |
--   +---- players
--            |
--            +---- stats (statistiques de saison)
--
-- matches
--   |
--   +---- reports
--
-- IMPORTANT :
-- Le fichier Excel "regular NBA.xlsx" contient des statistiques
-- de saison par joueur, et non des statistiques match par match.
-- Par conséquent, stats ne possède PAS de match_id.
-- ============================================================


-- ============================================================
-- 1. NETTOYAGE
-- ============================================================

DROP TABLE IF EXISTS reports CASCADE;
DROP TABLE IF EXISTS stats CASCADE;
DROP TABLE IF EXISTS matches CASCADE;
DROP TABLE IF EXISTS players CASCADE;
DROP TABLE IF EXISTS teams CASCADE;


-- ============================================================
-- 2. TABLE TEAMS
-- ============================================================

CREATE TABLE teams (
    code VARCHAR(3) PRIMARY KEY,

    name VARCHAR(100) NOT NULL UNIQUE
);


-- ============================================================
-- 3. TABLE PLAYERS
-- ============================================================
--
-- Cette table contient l'identité du joueur.
--
-- Les statistiques de saison sont stockées dans la table stats.
-- ============================================================

CREATE TABLE players (
    id INTEGER PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    team_code VARCHAR(3),
    age INTEGER,

    games_played INTEGER,
    wins INTEGER,
    losses INTEGER,
    minutes_per_game DOUBLE PRECISION,

    total_points DOUBLE PRECISION,
    points_per_game DOUBLE PRECISION,

    field_goals_made DOUBLE PRECISION,
    field_goals_attempted DOUBLE PRECISION,
    field_goal_pct DOUBLE PRECISION,

    three_pointers_made DOUBLE PRECISION,
    three_pointers_attempted DOUBLE PRECISION,
    three_point_pct DOUBLE PRECISION,

    free_throws_made DOUBLE PRECISION,
    free_throws_attempted DOUBLE PRECISION,
    free_throw_pct DOUBLE PRECISION,

    offensive_rebounds DOUBLE PRECISION,
    defensive_rebounds DOUBLE PRECISION,
    total_rebounds DOUBLE PRECISION,

    assists DOUBLE PRECISION,
    turnovers DOUBLE PRECISION,

    steals DOUBLE PRECISION,
    blocks DOUBLE PRECISION,
    personal_fouls DOUBLE PRECISION,

    fantasy_points DOUBLE PRECISION,

    double_doubles INTEGER,
    triple_doubles INTEGER,

    plus_minus DOUBLE PRECISION,

    offensive_rating DOUBLE PRECISION,
    defensive_rating DOUBLE PRECISION,
    net_rating DOUBLE PRECISION,

    assist_pct DOUBLE PRECISION,
    assist_to_turnover DOUBLE PRECISION,
    assist_ratio DOUBLE PRECISION,

    offensive_rebound_pct DOUBLE PRECISION,
    defensive_rebound_pct DOUBLE PRECISION,
    total_rebound_pct DOUBLE PRECISION,

    turnover_ratio DOUBLE PRECISION,

    effective_fg_pct DOUBLE PRECISION,
    true_shooting_pct DOUBLE PRECISION,

    usage_rate DOUBLE PRECISION,
    pace DOUBLE PRECISION,

    player_impact_estimate DOUBLE PRECISION,
    possessions DOUBLE PRECISION,

    CONSTRAINT fk_players_team
        FOREIGN KEY (team_code)
        REFERENCES teams(code)
        ON UPDATE CASCADE
        ON DELETE SET NULL,

    CONSTRAINT chk_players_age
        CHECK (age IS NULL OR age >= 0)
);


-- ============================================================
-- 4. TABLE STATS
-- ============================================================
--
-- Statistiques de SAISON provenant de regular NBA.xlsx
--
-- Une ligne = un joueur pour une saison.
--
-- Exemple :
--
-- player_id = 1
-- season = '2024-2025'
-- points = 2485
-- three_point_pct = 37.5
--
-- IMPORTANT :
-- Cette table ne contient PAS match_id car l'Excel fourni
-- ne contient pas les statistiques match par match.
-- ============================================================

CREATE TABLE stats (
    id BIGSERIAL PRIMARY KEY,

    player_id INTEGER NOT NULL,

    season VARCHAR(20),

    -- --------------------------------------------------------
    -- Matchs
    -- --------------------------------------------------------

    games_played INTEGER,

    wins INTEGER,

    losses INTEGER,


    -- --------------------------------------------------------
    -- Points
    -- --------------------------------------------------------

    points DOUBLE PRECISION,

    minutes_per_game DOUBLE PRECISION,

    total_points DOUBLE PRECISION,
    
    points_per_game DOUBLE PRECISION,


    -- --------------------------------------------------------
    -- Tirs
    -- --------------------------------------------------------

    field_goals_made DOUBLE PRECISION,

    field_goals_attempted DOUBLE PRECISION,

    field_goal_pct DOUBLE PRECISION,

    three_pointers_made DOUBLE PRECISION,

    three_pointers_attempted DOUBLE PRECISION,

    three_point_pct DOUBLE PRECISION,

    free_throws_made DOUBLE PRECISION,

    free_throws_attempted DOUBLE PRECISION,

    free_throw_pct DOUBLE PRECISION,


    -- --------------------------------------------------------
    -- Rebonds
    -- --------------------------------------------------------

    offensive_rebounds DOUBLE PRECISION,

    defensive_rebounds DOUBLE PRECISION,

    total_rebounds DOUBLE PRECISION,


    -- --------------------------------------------------------
    -- Passes / pertes de balle
    -- --------------------------------------------------------

    assists DOUBLE PRECISION,

    turnovers DOUBLE PRECISION,

    assist_pct DOUBLE PRECISION,

    assist_to_turnover DOUBLE PRECISION,

    assist_ratio DOUBLE PRECISION,

    turnover_ratio DOUBLE PRECISION,


    -- --------------------------------------------------------
    -- Défense
    -- --------------------------------------------------------

    steals DOUBLE PRECISION,

    blocks DOUBLE PRECISION,

    personal_fouls DOUBLE PRECISION,


    -- --------------------------------------------------------
    -- Statistiques avancées
    -- --------------------------------------------------------

    plus_minus DOUBLE PRECISION,

    offensive_rating DOUBLE PRECISION,

    defensive_rating DOUBLE PRECISION,

    net_rating DOUBLE PRECISION,

    offensive_rebound_pct DOUBLE PRECISION,

    defensive_rebound_pct DOUBLE PRECISION,

    total_rebound_pct DOUBLE PRECISION,

    effective_fg_pct DOUBLE PRECISION,

    true_shooting_pct DOUBLE PRECISION,

    usage_rate DOUBLE PRECISION,

    pace DOUBLE PRECISION,

    player_impact_estimate DOUBLE PRECISION,

    possessions DOUBLE PRECISION,


    -- --------------------------------------------------------
    -- Fantasy / performances
    -- --------------------------------------------------------

    fantasy_points DOUBLE PRECISION,

    double_doubles INTEGER,

    triple_doubles INTEGER,


    -- --------------------------------------------------------
    -- Contraintes
    -- --------------------------------------------------------

    CONSTRAINT fk_stats_player
        FOREIGN KEY (player_id)
        REFERENCES players(id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,

    CONSTRAINT uq_stats_player_season
        UNIQUE (player_id, season),


    -- --------------------------------------------------------
    -- Validations
    -- --------------------------------------------------------

    CONSTRAINT chk_stats_games_played
        CHECK (
            games_played IS NULL
            OR games_played >= 0
        ),

    CONSTRAINT chk_stats_wins
        CHECK (
            wins IS NULL
            OR wins >= 0
        ),

    CONSTRAINT chk_stats_losses
        CHECK (
            losses IS NULL
            OR losses >= 0
        ),

    CONSTRAINT chk_stats_points
        CHECK (
            points IS NULL
            OR points >= 0
        ),

    CONSTRAINT chk_stats_three_point_pct
        CHECK (
            three_point_pct IS NULL
            OR (
                three_point_pct >= 0
                AND three_point_pct <= 100
            )
        ),

    CONSTRAINT chk_stats_field_goal_pct
        CHECK (
            field_goal_pct IS NULL
            OR (
                field_goal_pct >= 0
                AND field_goal_pct <= 100
            )
        ),

    CONSTRAINT chk_stats_free_throw_pct
        CHECK (
            free_throw_pct IS NULL
            OR (
                free_throw_pct >= 0
                AND free_throw_pct <= 100
            )
        )
);


-- ============================================================
-- 5. TABLE MATCHES
-- ============================================================
--
-- Cette table est conservée pour permettre plus tard
-- l'intégration de données match par match.
--
-- Elle n'est PAS alimentée par le fichier Excel actuel.
-- ============================================================

CREATE TABLE matches (
    id BIGSERIAL PRIMARY KEY,

    match_date DATE,

    home_team_code VARCHAR(3),

    away_team_code VARCHAR(3),

    home_score INTEGER,

    away_score INTEGER,

    season VARCHAR(20),


    CONSTRAINT fk_matches_home_team
        FOREIGN KEY (home_team_code)
        REFERENCES teams(code)
        ON UPDATE CASCADE
        ON DELETE SET NULL,

    CONSTRAINT fk_matches_away_team
        FOREIGN KEY (away_team_code)
        REFERENCES teams(code)
        ON UPDATE CASCADE
        ON DELETE SET NULL,


    CONSTRAINT chk_matches_different_teams
        CHECK (
            home_team_code IS NULL
            OR away_team_code IS NULL
            OR home_team_code <> away_team_code
        ),


    CONSTRAINT chk_home_score
        CHECK (
            home_score IS NULL
            OR home_score >= 0
        ),


    CONSTRAINT chk_away_score
        CHECK (
            away_score IS NULL
            OR away_score >= 0
        )
);


-- ============================================================
-- 6. TABLE REPORTS
-- ============================================================
--
-- Rapports / articles / documents éventuellement associés
-- à un match.
--
-- Les PDF Reddit utilisés actuellement pour le RAG ne sont
-- PAS obligatoirement insérés ici.
-- Ils peuvent rester dans le pipeline vectoriel RAG.
-- ============================================================

CREATE TABLE reports (
    id BIGSERIAL PRIMARY KEY,

    match_id BIGINT,

    title VARCHAR(255),

    report_date DATE,

    content TEXT,

    source VARCHAR(255),

    url TEXT,


    CONSTRAINT fk_reports_match
        FOREIGN KEY (match_id)
        REFERENCES matches(id)
        ON UPDATE CASCADE
        ON DELETE SET NULL
);


-- ============================================================
-- 7. INDEX - PLAYERS
-- ============================================================

CREATE INDEX idx_players_team_code
    ON players(team_code);

CREATE INDEX idx_players_name
    ON players(name);


-- ============================================================
-- 8. INDEX - STATS
-- ============================================================

CREATE INDEX idx_stats_player
    ON stats(player_id);

CREATE INDEX idx_stats_season
    ON stats(season);

CREATE INDEX idx_stats_player_season
    ON stats(player_id, season);

CREATE INDEX idx_stats_points
    ON stats(points);

CREATE INDEX idx_stats_three_point_pct
    ON stats(three_point_pct);


-- ============================================================
-- 9. INDEX - MATCHES
-- ============================================================

CREATE INDEX idx_matches_date
    ON matches(match_date);

CREATE INDEX idx_matches_season
    ON matches(season);

CREATE INDEX idx_matches_home_team
    ON matches(home_team_code);

CREATE INDEX idx_matches_away_team
    ON matches(away_team_code);


-- ============================================================
-- 10. INDEX - REPORTS
-- ============================================================

CREATE INDEX idx_reports_match
    ON reports(match_id);

CREATE INDEX idx_reports_date
    ON reports(report_date);


-- ============================================================
-- FIN
-- ============================================================

SELECT 'Database schema created successfully.' AS status;