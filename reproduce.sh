#!/usr/bin/env bash
# Reproduction pipeline.   bash reproduce.sh quick   (~10 min: checks against the paper's logs, tables, figures)
#                           bash reproduce.sh full    (~70 CPU-hours: the whole study from scratch into results/)
# WORKERS=8 bash reproduce.sh full   runs 8 trainings in parallel (one CPU core each).
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-2}"
case "${1:-quick}" in
  quick)
    python scripts/env_report.py
    python -m unittest discover -s tests -v
    python scripts/make_data.py
    python scripts/check_paper.py --workers "$WORKERS"
    python scripts/analyze.py
    python scripts/latex_tables.py
    python scripts/make_figures.py --cd-sets abc
    ;;
  full)
    python scripts/env_report.py --save results_environment.json
    python scripts/tune_lr.py    --workers "$WORKERS" --results results
    python scripts/run_seeds.py  --workers "$WORKERS" --results results
    python scripts/run_curves.py --workers "$WORKERS" --results results
    python scripts/time_benchmark.py --results results
    python scripts/analyze.py      --results results
    python scripts/latex_tables.py --results results
    python scripts/make_figures.py --results results --cd-sets abc
    ;;
  *) echo "usage: bash reproduce.sh quick|full"; exit 1 ;;
esac
