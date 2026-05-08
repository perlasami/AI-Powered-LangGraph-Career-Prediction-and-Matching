# database/vector_store.py
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import json
import os
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client = chromadb.PersistentClient(
    path="database/chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_or_create_collection(name="jobs")


def build_text_for_embedding(job):
    return f"""
    Title: {job.get('title', '')}
    Company: {job.get('company', '')}
    Location: {job.get('location', '')}
    Description: {job.get('description', '')}
    Requirements: {job.get('requirements', '')}
    Job Type: {job.get('job_type', '')}
    Experience: {job.get('experience', '')}
    Level: {job.get('level', '')}
    """.strip()


def load_jobs_to_db():
    """Wuzzuf jobs"""
    with open("data/processed/wuzzuf_full.json", "r", encoding="utf-8") as f:
        jobs = json.load(f)

    print(f"📋 Loading {len(jobs)} Wuzzuf jobs...")
    _add_jobs(jobs, id_prefix="job")
    print(f"✅ Done! {collection.count()} total jobs in DB")


def load_linkedin_to_db():
    """LinkedIn jobs"""
    linkedin_path = "data/processed/linkedin_full.json"
    if not os.path.exists(linkedin_path):
        print("ℹ️  LinkedIn data not found, skipping")
        return

    with open(linkedin_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    existing = collection.get()
    existing_urls = {m['url'] for m in existing['metadatas']} if existing['metadatas'] else set()
    new_jobs = [j for j in jobs if j.get('url', '') not in existing_urls]

    if not new_jobs:
        print("ℹ️  All LinkedIn jobs already in DB")
        return

    print(f"📋 Loading {len(new_jobs)} new LinkedIn jobs...")
    _add_jobs(new_jobs, id_prefix="linkedin", start_index=collection.count())
    print(f"✅ Done! {collection.count()} total jobs in DB")

def load_nti_iti_to_db():
     """NTI/ITI programs"""
     nti_path = "data/raw/nti_iti_official.json"
     if not os.path.exists(nti_path):
        print("ℹ️  NTI/ITI data not found, skipping")
        return

     with open(nti_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)

     existing = collection.get()
     existing_urls = {m['url'] for m in existing['metadatas']} if existing['metadatas'] else set()
     new_jobs = [j for j in jobs if j.get('url', '') not in existing_urls]

     if not new_jobs:
        print("ℹ️  All NTI/ITI programs already in DB")
        return

     print(f"📋 Loading {len(new_jobs)} NTI/ITI programs...")
     _add_jobs(new_jobs, id_prefix="nti_iti", start_index=collection.count())
     print(f"✅ Done! {collection.count()} total jobs in DB") 
def _add_jobs(jobs, id_prefix="job", start_index=0):
    """Helper مشترك بين Wuzzuf و LinkedIn"""
    documents, embeddings, metadatas, ids = [], [], [], []

    for i, job in enumerate(jobs):
        text = build_text_for_embedding(job)
        documents.append(text)

        metadatas.append({
            "title":      job.get("title", ""),
            "company":    job.get("company", ""),
            "location":   job.get("location", ""),
            "job_type":   job.get("job_type", ""),
            "experience": job.get("experience", ""),
            "level":      job.get("level", ""),
            "url":        job.get("url", ""),
            "source":     job.get("source", "wuzzuf")  # ← عارف الوظيفة جت منين
        })

        ids.append(f"{id_prefix}_{start_index + i}")

    print("🔄 Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    




if __name__ == "__main__":
    # لو عايز تضيف Wuzzuf من الأول
    load_jobs_to_db()
    load_linkedin_to_db()
    load_nti_iti_to_db()


    
    print(f"\n📊 DB Summary:")
    all_meta = collection.get()['metadatas']
    wuzzuf_count   = sum(1 for m in all_meta if m.get('source') == 'wuzzuf')
    linkedin_count = sum(1 for m in all_meta if m.get('source') == 'linkedin')
    nti_iti_count  = sum(1 for m in all_meta if m.get('source') in ('nti', 'iti'))
    print(f"   Wuzzuf:   {wuzzuf_count}")
    print(f"   LinkedIn: {linkedin_count}")
    print(f"   NTI/ITI:  {nti_iti_count}")
    print(f"   Total:    {len(all_meta)}")