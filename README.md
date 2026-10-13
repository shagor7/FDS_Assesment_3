# FIFA World Cup 2026 — Task 2.2: Goals Scored Prediction

## Project overview
This repository contains the data preparation pipeline and analysis for Task 2.2: predicting the number of goals scored by a team in a FIFA World Cup 2026 match. The modelling dataset has one row per team per match (208 observations for 104 matches), with `goals_scored` as the target.

> **Data verification note:** The included source files and checks establish internal consistency between the local JSON/CSV and derived datasets; they do not independently authenticate every match result. Verify the match records against an authoritative source before relying on the results in an assessment.

## Repository contents
- `src/collect_and_prepare.py` — downloads/reads source data and builds the prepared datasets.
- `data/raw/` — source data files used by the project.
- `data/processed/regression_2_2.csv` — Task 2.2 modelling dataset.
- `data/processed/matches_with_pre_match_features_audit.csv` — audit data used to trace feature construction.
- `src/model_comparison.py` — cross-validated baseline and regression comparison.
- `results/` — generated metric tables and validation summary.
- `figures/` — charts generated from the dataset.

## Target and predictors
Target: `goals_scored`. The eight candidate predictors are `recent_gf_avg`, `recent_ga_avg`, `recent_win_rate`, `recent_goal_diff_avg`, `recent_points_avg`, `rest_days`, `host_team_indicator`, and `knockout_stage`. `match_id`, `date`, `team`, and `opponent` are identifiers/context fields, not model predictors.

The host indicator identifies Canada, Mexico, or the United States as a host nation; it is **not** a verified stadium-level home-advantage measure. Historical team features use matches dated strictly before the World Cup match date.

## Data sources
- OpenFootball World Cup JSON: https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json
- International match results: https://raw.githubusercontent.com/martj42/international_results/master/results.csv

See `DATA_SOURCES.md` for notes on provenance and limitations.

## Setup
Python 3.10 or newer is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To regenerate the datasets from the available sources:

```bash
python src/collect_and_prepare.py
```

To run Task 2.2 model comparison and generate results/figures:

```bash
python src/model_comparison.py
```

The modelling script uses grouped cross-validation so the two team observations from one match are kept in the same fold.

## Evaluation
The script compares a mean-prediction baseline, Linear Regression, and Ridge Regression using MAE, RMSE, and R-squared. Results are written to `results/model_comparison.csv` and `results/model_comparison.txt`. Exact results can vary if source datasets or package versions change.

## Limitations and checks before submission
1. Verify the match data against a reliable, authoritative source and cite the verification source.
2. Check that all eight predictors are known before the match and that the assignment's shared-predictor restriction is satisfied when considered alongside Task 2.1.
3. Historical results may have coverage or team-name mapping limitations.
4. Goals are count outcomes, while ordinary linear regression can predict non-integer or negative values; interpret predictions with this limitation in mind.
5. This repository is a reproducibility aid and does not replace the assessment report or required submission documents.
