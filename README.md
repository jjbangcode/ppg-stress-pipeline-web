# PPG Stress Classification Interactive Pipeline

Interactive local web application for WESAD-based PPG stress analysis.

## What It Does

- Connects to a local WESAD dataset directory.
- Runs batch PPG preprocessing across subjects.
- Supports standard harness parameters and custom preprocessing pipelines.
- Produces run-scoped artifacts under `result/{run_id}/`.
- Extracts Pulse-PPG embeddings and evaluates stress classification with XGBoost LOSO validation.

## Project Structure

```text
web/
├── index.html
├── style.css
├── app.js
├── server.py
└── run_server.sh
```

Generated analysis artifacts are intentionally excluded from git.

```text
result/
└── 0001_YYYYMMDD_HHMMSS/
    ├── data/
    ├── graphs/
    └── summaries/
```

## Run Locally

```bash
cd "/Users/su-younlee/Dropbox/PPG/1_PPG data analysis/web"
./run_server.sh
```

Then open:

```text
http://localhost:8050
```

## Notes

This project expects the local `ppg-stress` conda/miniforge environment used by `run_server.sh`.
