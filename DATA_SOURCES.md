# Data sources and provenance

## World Cup match data
- URL: https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json
- Local file: `data/raw/worldcup2026.json`
- Used to construct match dates, teams, round/stage, venue, and target scores.

## Historical international results
- URL: https://raw.githubusercontent.com/martj42/international_results/master/results.csv
- Local file: `data/raw/international_results.csv`
- Used to derive recent team goals, goals conceded, win rate, points, goal difference, and rest days.

## Important limitation
A match count of 104 and agreement between local JSON and CSV files are internal consistency checks. They do not prove that every match result is authentic or that the source file has complete coverage. Before submission, compare the scores and dates against an authoritative competition source, record the verification date, and cite that source in the report.
