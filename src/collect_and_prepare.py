"""
FIFA World Cup 2026 Objective 2 — reproducible data collector
Downloads public match results and historical international results,
then creates the required 104-row and 208-row modelling datasets.

Run:
    python -m pip install pandas requests
    python src/collect_and_prepare.py

Important:
- Verify source data and every row before using in an assessment.
- Historical match form is calculated strictly before each World Cup kickoff date.
- This script does NOT invent missing observations.
- It uses previous international match results; it does not provide historical
  shots-on-target or verified historical FIFA ranking points.
"""
from pathlib import Path
import json
import sys
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

WC_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
INTL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

def download(url, path):
    print(f"Downloading {url}")
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"Saved {path.relative_to(ROOT)} ({len(response.content):,} bytes)")

def read_world_cup():
    path = RAW / "worldcup2026.json"
    if not path.exists():
        download(WC_URL, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for i, m in enumerate(payload.get("matches", []), start=1):
        score = m.get("score") or {}
        # For target outcomes, use regulation/extra-time final score ('ft' in this
        # source represents the match score; do not add penalty-shootout scores).
        ft = score.get("ft")
        if not ft or len(ft) != 2 or ft[0] is None or ft[1] is None:
            continue
        rows.append({
            "match_id": i,
            "date": m.get("date"),
            "round": m.get("round", ""),
            "group": m.get("group", ""),
            "team_a": m.get("team1"),
            "team_b": m.get("team2"),
            "goals_a": int(ft[0]),
            "goals_b": int(ft[1]),
            "venue": m.get("ground", ""),
            "source_time": m.get("time", "")
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "team_a", "team_b", "goals_a", "goals_b"])
    # Exclude unresolved placeholder teams, if any remain in source.
    df = df[~df["team_a"].astype(str).str.match(r"^[0-9]+[A-LW]", na=False)]
    df = df[~df["team_b"].astype(str).str.match(r"^[0-9]+[A-LW]", na=False)]
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    df["goal_difference"] = df["goals_a"] - df["goals_b"]
    return df

def read_international():
    path = RAW / "international_results.csv"

    if not path.exists():
        download(INTL_URL, path)

    df = pd.read_csv(path)

    # Standardise historical names to match the World Cup dataset.
    TEAM_NAME_MAP = {
        "United States": "USA",
        "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    }

    for column in ["home_team", "away_team"]:
        df[column] = df[column].replace(TEAM_NAME_MAP)

    # Validate required columns.
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Historical results source is missing columns: {sorted(missing)}"
        )

    # Parse dates and scores.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(
        df["home_score"], errors="coerce"
    )
    df["away_score"] = pd.to_numeric(
        df["away_score"], errors="coerce"
    )

    # Remove invalid observations.
    df = df.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ).copy()

    # Keep only non-negative integer scores.
    valid_scores = (
        df["home_score"].ge(0)
        & df["away_score"].ge(0)
        & df["home_score"].mod(1).eq(0)
        & df["away_score"].mod(1).eq(0)
    )

    df = df.loc[valid_scores].copy()

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Create one historical record for each participating team.
    records = []

    for r in df.itertuples(index=False):
        hs = int(r.home_score)
        aws = int(r.away_score)

        records.append({
            "date": r.date,
            "team": r.home_team,
            "opponent": r.away_team,
            "gf": hs,
            "ga": aws,
            "win": int(hs > aws),
            "points": 3 if hs > aws else (1 if hs == aws else 0),
        })

        records.append({
            "date": r.date,
            "team": r.away_team,
            "opponent": r.home_team,
            "gf": aws,
            "ga": hs,
            "win": int(aws > hs),
            "points": 3 if aws > hs else (1 if hs == aws else 0),
        })

    history = pd.DataFrame(records)

    return history.sort_values(
        ["date", "team"]
    ).reset_index(drop=True)

def team_history_features(history, team, cutoff, n=5):
    # Strictly earlier dates only, avoiding use of same-day outcomes when exact
    # kickoff timestamps are not consistently available in the historical source.
    prior = history[(history["team"] == team) & (history["date"] < cutoff)].tail(n)
    if prior.empty:
        return {
            "recent_gf_avg": None, "recent_ga_avg": None,
            "recent_win_rate": None, "recent_points_avg": None,
            "recent_goal_diff_avg": None, "rest_days": None
        }
    previous_date = prior["date"].max()
    return {
        "recent_gf_avg": prior["gf"].mean(),
        "recent_ga_avg": prior["ga"].mean(),
        "recent_win_rate": prior["win"].mean(),
        "recent_points_avg": prior["points"].mean(),
        "recent_goal_diff_avg": (prior["gf"] - prior["ga"]).mean(),
        "rest_days": (cutoff - previous_date).days
    }

def is_host(team, venue):
    # Simple host-country proxy. Team names are normalised for common naming.
    t = str(team).strip().lower()
    v = str(venue).strip().lower()
    if t in {"canada", "mexico", "usa", "united states"}:
        return 1
    return 0

def main():
    matches = read_world_cup()
    history = read_international()
    matches.to_csv(RAW / "world_cup_matches_downloaded.csv", index=False)
    print(f"World Cup rows with scores: {len(matches)}")
    if len(matches) != 104:
        print("WARNING: Expected 104 completed matches; source returned a different count.")
        print("Check the source file, unresolved placeholders, and score fields before proceeding.")
    if matches[["team_a", "team_b", "date"]].isna().any().any():
        raise ValueError("Missing team/date values in World Cup data.")

    rows = []
    for m in matches.itertuples(index=False):
        cutoff = m.date
        a = team_history_features(history, m.team_a, cutoff)
        b = team_history_features(history, m.team_b, cutoff)
        stage_text = str(m.round).lower()
        knockout = int(any(x in stage_text for x in
                           ["round of", "quarter", "semi", "final", "third place"]))
        # The host flag is defined at national-team level; venue is not assumed
        # to mean the national team is playing at home.
        host_a, host_b = is_host(m.team_a, m.venue), is_host(m.team_b, m.venue)
        rows.append({
            **m._asdict(),
            "a_recent_gf_avg": a["recent_gf_avg"], "b_recent_gf_avg": b["recent_gf_avg"],
            "a_recent_ga_avg": a["recent_ga_avg"], "b_recent_ga_avg": b["recent_ga_avg"],
            "a_recent_win_rate": a["recent_win_rate"], "b_recent_win_rate": b["recent_win_rate"],
            "a_recent_points_avg": a["recent_points_avg"], "b_recent_points_avg": b["recent_points_avg"],
            "a_recent_goal_diff_avg": a["recent_goal_diff_avg"], "b_recent_goal_diff_avg": b["recent_goal_diff_avg"],
            "a_rest_days": a["rest_days"], "b_rest_days": b["rest_days"],
            "host_a": host_a, "host_b": host_b, "knockout_stage": knockout
        })
    enriched = pd.DataFrame(rows)

    # Model 2.1: exactly eight predictors; four difference features plus rest,
    # host, stage, and recent points difference.
    model21 = pd.DataFrame({
        "match_id": enriched["match_id"],
        "date": enriched["date"],
        "team_a": enriched["team_a"],
        "team_b": enriched["team_b"],
        "recent_gf_avg_diff": enriched["a_recent_gf_avg"] - enriched["b_recent_gf_avg"],
        "recent_ga_avg_diff_reversed": enriched["b_recent_ga_avg"] - enriched["a_recent_ga_avg"],
        "recent_win_rate_diff": enriched["a_recent_win_rate"] - enriched["b_recent_win_rate"],
        "recent_goal_diff_avg_diff": enriched["a_recent_goal_diff_avg"] - enriched["b_recent_goal_diff_avg"],
        "rest_days_diff": enriched["a_rest_days"] - enriched["b_rest_days"],
        "host_advantage_diff": enriched["host_a"] - enriched["host_b"],
        "knockout_stage": enriched["knockout_stage"],
        "recent_points_avg_diff": enriched["a_recent_points_avg"] - enriched["b_recent_points_avg"],
        "goal_difference": enriched["goal_difference"]
    })

    # Model 2.2: one row per team per match. Four concepts shared with 2.1:
    # recent GF, recent GA, recent win rate, and recent goal difference.
    team_rows = []
    for m in enriched.itertuples(index=False):
        for side in ("a", "b"):
            team_rows.append({
                "match_id": m.match_id, "date": m.date,
                "team": getattr(m, f"team_{side}"),
                "opponent": getattr(m, "team_b" if side == "a" else "team_a"),
                "recent_gf_avg": getattr(m, f"{side}_recent_gf_avg"),
                "recent_ga_avg": getattr(m, f"{side}_recent_ga_avg"),
                "recent_win_rate": getattr(m, f"{side}_recent_win_rate"),
                "recent_goal_diff_avg": getattr(m, f"{side}_recent_goal_diff_avg"),
                "recent_points_avg": getattr(m, f"{side}_recent_points_avg"),
                "rest_days": getattr(m, f"{side}_rest_days"),
                "host_team_indicator": getattr(m, f"host_{side}"),
                "knockout_stage": m.knockout_stage,
                "goals_scored": getattr(m, f"goals_{side}")
            })
    model22 = pd.DataFrame(team_rows)

    model21.to_csv(OUT / "regression_2_1.csv", index=False)
    model22.to_csv(OUT / "regression_2_2.csv", index=False)
    enriched.to_csv(OUT / "matches_with_pre_match_features_audit.csv", index=False)

    print("\n--- Validation summary ---")
    print("Model 2.1 rows:", len(model21), "| expected 104")
    print("Model 2.1 predictors:", 8)
    print("Model 2.2 rows:", len(model22), "| expected 208")
    print("Model 2.2 predictors:", 8)
    print("Model 2.1 missing values in predictors:")
    print(model21.iloc[:, 4:12].isna().sum())
    print("Model 2.2 missing values in predictors:")
    print(model22.iloc[:, 4:12].isna().sum())
    print("\nFiles written to:", OUT)
    print("Review missing values, host coding, stage coding, source provenance, and score rules before modelling.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
