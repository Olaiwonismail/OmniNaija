#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
python scripts/amazon_ingest.py --per-category 2000
python scripts/build_chroma.py --persist-dir chroma_db
python scripts/sanity_query.py --persist-dir chroma_db
