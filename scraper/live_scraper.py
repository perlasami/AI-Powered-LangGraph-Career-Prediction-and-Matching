import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.stdout.reconfigure(encoding='utf-8')

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import urllib.parse, time, json
from concurrent.futures import ThreadPoolExecutor
from database.pinecone_retriever import is_job_fresh

BASE_URL = "https://wuzzuf.net"

# ===============================
# Ensure folders exist
# ===============================
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)


# ===============================
# GET JOB DETAILS
# ===============================
def get_job_details_sync(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1500)

            soup = BeautifulSoup(page.content(), 'lxml')

            desc = soup.find('div', class_='css-1lqavbg')
            skills = soup.find('div', class_='css-14zw0ku')
            job_type = soup.find('a', class_='css-g65o95')
            work_style = soup.find('a', class_='css-wzyv7i')
            exp_spans = soup.find_all('span', class_='css-iu2m7n')

            browser.close()

            return {
                "description": desc.get_text(separator='\n', strip=True) if desc else "N/A",
                "requirements": skills.get_text(separator='\n', strip=True) if skills else "N/A",
                "job_type": job_type.get_text(strip=True) if job_type else "N/A",
                "work_style": work_style.get_text(strip=True) if work_style else "N/A",
                "experience": exp_spans[0].get_text(strip=True) if exp_spans else "N/A",
                "level": exp_spans[1].get_text(strip=True) if len(exp_spans) > 1 else "N/A",
            }

    except Exception as e:
        print(f"❌ Details error: {e}")
        return {}


# ===============================
# MAIN SCRAPER
# ===============================
def _scrape_sync(query, num_pages=2):
    print(f"🌐 Live scraping: '{query}'...")

    encoded = urllib.parse.quote(query)
    jobs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for page_num in range(num_pages):
            url = f"{BASE_URL}/search/jobs/?q={encoded}&a=1&filters[country][0]=Egypt&start={page_num * 15}"
            page.goto(url)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(2000)

            soup = BeautifulSoup(page.content(), 'lxml')
            job_links = soup.find_all('a', href=lambda x: x and '/jobs/p/' in x)

            for link in job_links:
                card = link.find_parent('div')
                if not card:
                    continue

                title = link.get_text(strip=True)
                job_url = BASE_URL + link['href'] if link['href'].startswith('/') else link['href']

                company_tag = card.find('a')
                company = company_tag.get_text(strip=True) if company_tag else "N/A"

                location_tag = card.find('span')
                location = location_tag.get_text(strip=True) if location_tag else "N/A"

                job_obj = {
                    "title": title,
                    "url": job_url,
                    "company": company,
                    "location": location,
                    "posted": "N/A",
                    "job_type": "N/A",
                    "experience": "N/A",
                    "level": "N/A",
                    "description": "N/A",
                    "requirements": "N/A",
                    "source": "live"
                }

                # Skip old jobs safely
                try:
                    if not is_job_fresh(job_obj):
                        print(f"⏭️ Skipping old job: {title}")
                        continue
                except:
                    pass

                jobs.append(job_obj)

            time.sleep(1.5)

        browser.close()

    print(f"✅ Found {len(jobs)} fresh jobs")

    # ===============================
    # GET DETAILS
    # ===============================
    print("📋 Getting details...")

    for job in jobs:
        job.update(get_job_details_sync(job['url']))
        time.sleep(1)

    # ===============================
    # NTI / ITI (SAFE FIXED VERSION)
    # ===============================
    try:
        if any(w in query.lower() for w in ["internship", "training", "iti", "nti"]):

            nti_path = "data/raw/nti_iti_official.json"

            if os.path.exists(nti_path):
                print("🎓 Adding NTI/ITI programs...")

                with open(nti_path, "r", encoding="utf-8") as f:
                    nti_jobs = json.load(f)

                jobs.extend(nti_jobs)

    except Exception as e:
        print(f"❌ NTI/ITI error: {e}")

    # ===============================
    # REMOVE DUPLICATES
    # ===============================
    seen = set()
    unique_jobs = []

    for job in jobs:
        if job["url"] not in seen:
            seen.add(job["url"])
            unique_jobs.append(job)

    return unique_jobs


# ===============================
# ASYNC WRAPPER
# ===============================
async def live_search_async(query, num_pages=2):
    import asyncio
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as pool:
        jobs = await loop.run_in_executor(pool, _scrape_sync, query, num_pages)

    return jobs


# ===============================
# DB FUNCTION (PINECONE)
# ===============================
def add_to_db(new_jobs):
    import hashlib
    from dotenv import load_dotenv
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer

    load_dotenv()

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "jobs-rag-egypt"

    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY not found in .env — cannot save jobs.")
        return

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # ── Deduplicate against existing Pinecone records ──
    # Use URL hash as deterministic ID so we never upsert duplicates
    def url_to_id(url):
        return "live_" + hashlib.md5(url.encode()).hexdigest()[:16]

    candidate_ids = [url_to_id(j['url']) for j in new_jobs]

    # Fetch existing IDs from Pinecone (batch fetch in groups of 100)
    existing_ids = set()
    for i in range(0, len(candidate_ids), 100):
        batch = candidate_ids[i:i+100]
        try:
            fetch_result = index.fetch(ids=batch)
            existing_ids.update(fetch_result.vectors.keys())
        except Exception as e:
            print(f"⚠️ Fetch check failed: {e}")

    truly_new = [
        j for j, vid in zip(new_jobs, candidate_ids)
        if vid not in existing_ids
    ]

    if not truly_new:
        print("ℹ️ All jobs already in Pinecone")
        return

    print(f"➕ Adding {len(truly_new)} new jobs to Pinecone...")

    # ── Build vectors ──
    vectors = []
    for job in truly_new:
        doc_text = (
            f"Title: {job.get('title', '')} "
            f"Company: {job.get('company', '')} "
            f"Location: {job.get('location', '')} "
            f"Description: {job.get('description', '')[:300]}"
        )
        embedding = model.encode(doc_text).tolist()

        # Pinecone metadata values must be str/int/float/bool/list — sanitize
        metadata = {
            k: (str(v) if v is not None else "N/A")
            for k, v in job.items()
            if isinstance(v, (str, int, float, bool)) or v is None
        }

        vectors.append({
            "id":       url_to_id(job['url']),
            "values":   embedding,
            "metadata": metadata,
        })

    # ── Upsert in batches of 100 ──
    try:
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i+100]
            index.upsert(vectors=batch)
            print(f"   ✅ Upserted batch {i//100 + 1} ({len(batch)} records)")
        print(f"✅ Pinecone updated! Added {len(vectors)} new jobs.")
    except Exception as e:
        print(f"❌ Pinecone upsert failed: {e}")
        return

    # ── Also update local JSON ──
    json_path = "data/processed/wuzzuf_full.json"

    if not os.path.exists(json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    with open(json_path, "r", encoding="utf-8") as f:
        all_jobs = json.load(f)

    all_jobs.extend(truly_new)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, ensure_ascii=False, indent=2)

    print(f"✅ Local JSON updated too.")

def save_to_json(jobs):
    path = "data/processed/wuzzuf_full.json"

    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f)

    with open(path, "r", encoding="utf-8") as f:
        existing = json.load(f)

    existing.extend(jobs)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
# ===============================
# TEST RUN ONLY
# ===============================
if __name__ == "__main__":
    import asyncio

    results = asyncio.run(live_search_async("python"))
    save_to_json(results) 
    print(f"Jobs found: {len(results)}")