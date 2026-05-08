# database/bm25_index.py
from rank_bm25 import BM25Okapi
import json, pickle, os
import os
import json

def build_bm25_index():
    wuzzuf_path = "data/processed/wuzzuf_full.json"
    linkedin_path = "data/processed/linkedin_full.json"
    nti_iti_path = "data/raw/nti_iti_official.json"  # ← ADD THIS

    wuzzuf_jobs = []
    if os.path.exists(wuzzuf_path):
        with open(wuzzuf_path, "r", encoding="utf-8") as f:
            wuzzuf_jobs = json.load(f)
    else:
        print("⚠️ Wuzzuf dataset not found.")

    linkedin_jobs = []
    if os.path.exists(linkedin_path):
        with open(linkedin_path, "r", encoding="utf-8") as f:
            linkedin_jobs = json.load(f)
    else:
        print("ℹ️  LinkedIn data not found, skipping")

    nti_iti_jobs = []                                  # ← ADD THIS
    if os.path.exists(nti_iti_path):
        with open(nti_iti_path, "r", encoding="utf-8") as f:
            nti_iti_jobs = json.load(f)
        print(f"✅ Loaded {len(nti_iti_jobs)} NTI/ITI programs")
    else:
        print("ℹ️  NTI/ITI data not found, skipping")

    all_jobs = wuzzuf_jobs + linkedin_jobs + nti_iti_jobs  # ← ADD nti_iti_jobs

    print(f"📋 Building BM25 index for {len(all_jobs)} jobs...")
    print(f"   Wuzzuf:   {len(wuzzuf_jobs)}")
    print(f"   LinkedIn: {len(linkedin_jobs)}")
    print(f"   NTI/ITI:  {len(nti_iti_jobs)}")
    # ... rest unchanged


if __name__ == "__main__":
    build_bm25_index()