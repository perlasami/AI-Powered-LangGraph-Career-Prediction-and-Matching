# database/pinecone_retriever.py
"""
Pinecone-based hybrid search retriever
Combines vector search (Pinecone) + BM25 (local) using RRF
"""

import os
import pickle
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "jobs-rag-egypt"

# Initialize
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
else:
    print("⚠️  PINECONE_API_KEY not found. Using fallback to Chroma.")
    import chromadb
    index = None

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load BM25 index (keeping it local for fast keyword search)
try:
    with open("database/bm25_index.pkl", "rb") as f:
        data = pickle.load(f)
        bm25 = data["bm25"]
        jobs = data["jobs"]
except FileNotFoundError:
    print("⚠️  BM25 index not found. Vector-only search will be used.")
    bm25 = None
    jobs = {}

def is_job_fresh(job_dict):
    """
    Check if a job is recent (within 1 month).
    Filters out jobs with 'month', 'months', 'year', 'years' in posted date.
    Keeps jobs with 'day', 'days', 'week', 'weeks', 'hour', 'hours', 'minute', 'minutes' or 'recently', 'now'.
    Also checks 'time' or 'date' keys if 'posted' is missing.
    
    Args:
        job_dict: Job dictionary with posted/time/date field
        
    Returns:
        bool: True if job is fresh (< 1 month), False otherwise
    """
    posted_val = job_dict.get('posted') or job_dict.get('time') or job_dict.get('date') or "N/A"
    posted_str = str(posted_val).lower().strip()
    
    # Filter out old jobs
    if any(old_marker in posted_str for old_marker in ['month', 'months', 'year', 'years']):
        return False
    
    # Accept fresh indicators
    if any(fresh_marker in posted_str for fresh_marker in ['day', 'days', 'week', 'weeks', 'hour', 'hours', 'minute', 'minutes', 'recently', 'now', 'today']):
        return True
    
    # Default: keep (if we can't determine, better to include)
    return True


def hybrid_search(query, n_results=15, sources=None):
    """
    Hybrid search combining:
    1. Vector search (Pinecone) - semantic similarity
    2. BM25 (local) - keyword matching
    3. RRF (Reciprocal Rank Fusion) - combine both
    """
    print(f"\n🔍 Query: '{query}'")
    print("=" * 50)

    rrf_scores = {}
    all_jobs_by_url = {job['url']: job for job in jobs}
    
    K = 60  # RRF parameter

    # ── 1. Vector Search (Pinecone) ──
    if index:
        print("📊 Searching Pinecone vectors...")
        query_vector = model.encode(query).tolist()
        
        vector_results = index.query(
            vector=query_vector,
            top_k=20,
            include_metadata=True
        )
        
        for rank, match in enumerate(vector_results.matches):
            if not is_job_fresh(match.metadata):
                continue
            
            url = match.metadata.get('url', match.id)
            score = 1 / (K + rank + 1)
            rrf_scores[url] = rrf_scores.get(url, 0) + score
            print(f"   #{rank + 1}: {match.metadata.get('title', 'N/A')} (score: {match.score:.4f})")
    
    # ── 2. BM25 Search (Local) ──
    if bm25:
        print("\n📚 Searching BM25 index...")
        query_words = query.lower().split()
        bm25_scores = bm25.get_scores(query_words)
        
        bm25_ranked = sorted(range(len(bm25_scores)), 
                           key=lambda i: bm25_scores[i], 
                           reverse=True)[:20]
        
        for rank, job_idx in enumerate(bm25_ranked):
            if job_idx < len(jobs):
                job = jobs[job_idx]
                if not is_job_fresh(job):
                    continue
                url = job['url']
                score = 1 / (K + rank + 1)
                rrf_scores[url] = rrf_scores.get(url, 0) + score
                print(f"   #{rank + 1}: {job.get('title', 'N/A')} (BM25: {bm25_scores[job_idx]:.4f})")
    
    # ── 3. RRF Ranking & Deduplication ──
    print("\n🔄 Combining results with RRF...")
    seen_titles = set()
    top_results = []

    # Normalize sources filter once
    allowed_sources = None
    if sources:
        allowed_sources = {s.lower().strip() for s in sources}
        if 'wuzzuf' in allowed_sources:
            allowed_sources.add('live')  # treat live scrape as wuzzuf alias

    for url in sorted(rrf_scores, key=rrf_scores.get, reverse=True):
        job = all_jobs_by_url.get(url)
        
        if not job:
            continue

        # Source filtering
        if allowed_sources:
            job_source = job.get('source', 'wuzzuf').lower()
            if job_source not in allowed_sources:
                continue
        
        title_key = job['title'].lower().strip()
        if title_key in seen_titles:
            continue
        
        seen_titles.add(title_key)
        top_results.append({
            "job": job,
            "score": round(rrf_scores[url], 4)
        })
        
        if len(top_results) == n_results:
            break
    
    # ── 4. Display Results ──
    print(f"\n✅ Top {len(top_results)} results:")
    for i, r in enumerate(top_results):
        job = r['job']
        print(f"\n#{i+1} (RRF score: {r['score']})")
        print(f"  📌 {job.get('title', 'N/A')}")
        print(f"  🏢 {job.get('company', 'N/A')}")
        print(f"  📍 {job.get('location', 'N/A')}")
        print(f"  🔗 {job.get('url', 'N/A')[:50]}...")
    
    return top_results


def simple_vector_search(query, n_results=10):
    """
    Simple vector-only search (if BM25 not available)
    """
    if not index:
        print("❌ Pinecone not configured")
        return []
    
    query_vector = model.encode(query).tolist()
    
    results = index.query(
        vector=query_vector,
        top_k=n_results,
        include_metadata=True
    )
    
    formatted_results = []
    for match in results.matches:
        if not is_job_fresh(match.metadata):
            continue
        formatted_results.append({
            "job": match.metadata,
            "score": round(match.score, 4)
        })
    
    return formatted_results


# Quick test
if __name__ == "__main__":
    test_query = "python developer"
    results = hybrid_search(test_query, n_results=5)
    print(f"\n📋 Found {len(results)} results")