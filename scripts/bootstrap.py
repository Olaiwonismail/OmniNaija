#!/usr/bin/env python3
"""
OmniNaija Lite — Clean PC Ingestion & Database Bootstrap Script
==============================================================
This script automates the complete data preparation and database construction
process so that OmniNaija can be fully run and evaluated from a clean PC clone.

It executes the following steps:
1. Generates 160+ synthetic Nigerian venues (Lagos/Abuja) with backup power facilities.
2. Ingests and processes 2,000 Amazon products per category (Electronics, Books, Home & Kitchen)
   by streaming from McAuley-Lab/Amazon-Reviews-2023 on Hugging Face (reviews-first flow).
3. Builds the persistent local vector database (ChromaDB) for both products and locations.
4. Generates the synthetic user interactions dataset needed for offline evaluation.
5. Runs the diagnostic health check to verify everything is 100% operational.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    print(f"\n==================================================")
    print(f"-> {description}")
    print(f"Running: {' '.join(command)}")
    print(f"==================================================")
    try:
        # Run with current python executable to ensure same virtual environment is used
        result = subprocess.run(command, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as err:
        print(f"\n[ERROR] Error during: {description}")
        print(err)
        sys.exit(1)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)

    print("==================================================")
    print("      OmniNaija Lite Database & Data Ingest       ")
    print("==================================================")
    print("This bootstrap script will prepare all local data and vector databases")
    print("so you can run evaluation and run the Streamlit app on a clean machine.\n")

    # Step 1: Generate Nigerian venues JSONL
    run_command(
        [sys.executable, "scripts/generate_venues.py"],
        "Step 1: Generating synthetic Nigerian venues dataset"
    )

    # Step 2: Build venue ChromaDB collection
    run_command(
        [sys.executable, "scripts/build_venue_chroma.py"],
        "Step 2: Building venue_locations collection in ChromaDB"
    )

    # Step 3: Stream and ingest Amazon reviews
    print("\n[WARNING] Note: The next step streams data directly from Hugging Face.")
    print("It uses an efficient reviews-first streaming strategy, so only a tiny")
    print("fraction of the dataset is processed. However, a stable internet connection is required.")
    run_command(
        [sys.executable, "scripts/amazon_ingest.py", "--per-category", "2000"],
        "Step 3: Streaming & Ingesting Amazon reviews from Hugging Face"
    )

    # Step 4: Build Amazon products ChromaDB collection
    run_command(
        [sys.executable, "scripts/build_chroma.py", "--persist-dir", "chroma_db"],
        "Step 4: Building amazon_products collection in ChromaDB"
    )

    # Step 5: Build synthetic user interactions for offline evaluation
    run_command(
        [sys.executable, "scripts/build_interactions.py"],
        "Step 5: Generating user_interactions.jsonl for evaluation"
    )

    # Step 6: Diagnostic Health Check
    run_command(
        [sys.executable, "scripts/health_check.py"],
        "Step 6: Running diagnostic health checks"
    )

    print("\n[SUCCESS] BOOTSTRAP COMPLETE! OmniNaija Lite is now fully operational.")
    print("\n-> To run the live frontend and API in Docker:")
    print("   docker-compose up --build")
    print("\n-> To reproduce the quantitative evaluation metrics locally:")
    print("   python evaluation/evaluate_v2.py")

    print("==================================================")


if __name__ == "__main__":
    main()
