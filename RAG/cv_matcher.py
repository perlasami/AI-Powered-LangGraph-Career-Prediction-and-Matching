import fitz 
import math 
from sentence_transformers import SentenceTransformer 
from groq import Groq 
import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from dotenv import load_dotenv 
# fresh jobs date filter
from datetime import datetime, timedelta
# dates parsing
import re
# live scrapping
from scraper.live_scraper import live_search_async, add_to_db

load_dotenv() 

# Initialize Groq client with API key
client = Groq(api_key=os.getenv("GROQ_API_KEY")) 
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') 

# Initialize Pinecone (with fallback to Chroma)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "pinecone").lower()

if VECTOR_DB_TYPE == "pinecone" and PINECONE_API_KEY:
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "jobs-rag-egypt"))
        use_pinecone = True
        print("✅ Using Pinecone vector database")
    except Exception as e:
        print(f"⚠️  Pinecone failed: {e}, falling back to Chroma")
        import chromadb
        from chromadb.config import Settings
        db_client = chromadb.PersistentClient(
            path="database/chroma_db",
            settings=Settings(anonymized_telemetry=False)
        ) 
        collection = db_client.get_collection("jobs")
        use_pinecone = False
else:
    import chromadb
    from chromadb.config import Settings
    db_client = chromadb.PersistentClient(
        path="database/chroma_db",
        settings=Settings(anonymized_telemetry=False)
    ) 
    collection = db_client.get_collection("jobs")
    use_pinecone = False 

# Helper function to parse "posted date" strings into actual dates
def parse_posted_date(posted_str):
    """
    تحول النص زي '1 month ago' إلى تاريخ فعلي.
    """
    today = datetime.now()
    match = re.match(r'(\d+)\s+(day|days|month|months|year|years)\s+ago', posted_str)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if 'day' in unit:
            return today - timedelta(days=value)
        elif 'month' in unit:
            return today - timedelta(days=30 * value)
        elif 'year' in unit:
            return today - timedelta(days=365 * value)
    return None

# Fresh and open jobs filter
def is_recent_and_open(job):
    """فلترة الوظائف: افتح فقط والحديثة (آخر أسبوعين)"""
    status = job.get('status', '').lower()
    posted_str = job.get('posted', '')
    
    # تحقق من الحالة أولاً
    if status and status != 'open':
        return False
    
    if posted_str:
        # محاولة parse الصيغة "X days/months ago"
        posted_date = parse_posted_date(posted_str)
        if posted_date:
            today = datetime.now()
            if (today - posted_date) <= timedelta(days=30):  # آخر أسبوعين
                return True
        # محاولة parse الصيغة YYYY-MM-DD
        try:
            posted_date = datetime.strptime(posted_str, "%Y-%m-%d")
            today = datetime.now()
            if (today - posted_date) <= timedelta(days=30):
                return True
        except:
            pass
    return False

def extract_text_from_pdf(pdf_bytes): 
    """Extracts text from uploaded PDF bytes."""
    try:
        if not pdf_bytes:
            return None
        doc = fitz.open(stream=pdf_bytes, filetype="pdf") 
        if doc.page_count == 0:
            return None
        text = "" 
        for page in doc: 
            text += page.get_text() 
        doc.close()
        return text.strip() if text.strip() else None
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")
        return None 

def extract_skills_from_cv(cv_text): 
    """
    Uses Groq LLM to extract key skills, technologies, and experience 
    from the full CV text to optimize vector embeddings.
    """
    prompt = f"""
    Analyze the following resume text and extract the key technical skills, 
    job titles, and years of experience. Provide a concise summary 
    suitable for job matching.
    
    Resume Text:
    {cv_text}
    """
    
    response = client.chat.completions.create( 
        model="llama-3.1-8b-instant", 
        messages=[{
            "role": "user",
            "content": f"""من الـ CV ده، استخرج:
1. المسمى الوظيفي المناسب
2. المهارات التقنية
3. سنوات الخبرة
4. المجال

رد بالإنجليزي فقط في صيغة:
Job Title: ...
Skills: skill1, skill2, skill3, ...
Experience: X years
Field: ...

CV:
{cv_text[:3000]}"""
        }],
        max_tokens=200
    )

    extracted = response.choices[0].message.content
    print(f"📋 Extracted profile:\n{extracted}\n")
    return extracted

# تحويل المسافة إلى درجة مطابقة (0-100)
def distance_to_score(distance):
    return round(math.exp(-distance / 10) * 100, 1)


def build_search_query(skills_profile: str) -> str:
    # استخرج Job Title
    title_match = re.search(r'Job Title:\s*(.*)', skills_profile)
    title = title_match.group(1) if title_match else ""

    # استخرج Skills
    skills_match = re.search(r'Skills:\s*(.*)', skills_profile)
    skills = skills_match.group(1) if skills_match else ""

    # خد أهم 5 skills بس
    skills_list = [s.strip() for s in skills.split(",")][:5]

    query = f"{title} {' '.join(skills_list)}"
    
    return query.strip()

# Main CV matching function
async def match_cv_to_jobs(pdf_bytes, n_results=5):
    # ── 1. استخرج النص ──
    cv_text = extract_text_from_pdf(pdf_bytes)
    if not cv_text:
        return [], "مش قادر أقرأ الـ CV"

    print(f"📄 CV text: {len(cv_text)} chars")

    # ── 2. استخرج الـ skills بالـ LLM ──
    skills_profile = extract_skills_from_cv(cv_text)

    # ── 3. حول الـ profile للـ vector (مش الـ CV كامل) ──
    cv_vector = model.encode(skills_profile).tolist()

    # ── 4. دور في الـ DB (Pinecone or Chroma) ──
    if use_pinecone:
        # Query Pinecone
        results = pinecone_index.query(
            vector=cv_vector,
            top_k=n_results,
            include_metadata=True
        )
        
        distances = []
        metadatas = []
        for match in results.matches:
            # Pinecone returns similarity scores (0-1), convert to distance
            distance = 1 - match.score
            distances.append(distance)
            metadatas.append(match.metadata)
    else:
        # Query Chroma
        results = collection.query(
            query_embeddings=[cv_vector],
            n_results=n_results
        )
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

    print(f"Raw distances: {[round(d,2) for d in distances]}")

    # ── 5. احسب الـ scores ──
    matches = []
    for i, metadata in enumerate(metadatas):
        # تحويل المسافة إلى درجة مطابقة (0-100)
        score = distance_to_score(distances[i]) if i < len(distances) else 0
        matches.append({
            "title":      metadata.get("title", "N/A"),
            "company":    metadata.get("company", "N/A"),
            "location":   metadata.get("location", "N/A"),
            "experience": metadata.get("experience", "N/A"),
            "level":      metadata.get("level", "N/A"),
            "job_type":   metadata.get("job_type", "N/A"),
            "url":        metadata.get("url", "#"),
            "source":     metadata.get("source", "wuzzuf"),
            "posted":     metadata.get("posted", None),
            "match":      score  # إضافة درجة المطابقة
        })


    # ── 6. فلترة النتائج بعد الحسابات ──
    matches = [m for m in matches if is_recent_and_open(m)]

    print(f"Number of matches after filter: {len(matches)}")

    # فلترة الوظائف بناءً على إن كانت موجودة (posted) أم لا
    if not matches:
        from scraper.live_scraper import live_search_async, add_to_db
        import asyncio
        print("⚡️ Live scraping...")

        # نفذ الـ live scraping بناءً على نفس query الـ CV
        query = build_search_query(skills_profile)
        print(f"🔍 Search Query: {query}")

        live_jobs = await live_search_async(query)  # أو ممكن تعدل بناءً على الـ query نفسه
        
        if live_jobs:
            add_to_db(live_jobs)  # احفظهم في الـ DB
            # حوّل النتائج الجديدة إلى نفس الشكل
            matches = []
            for job in live_jobs[:n_results]:
                matches.append({
                    "title":      job.get("title", "N/A"),
                    "company":    job.get("company", "N/A"),
                    "location":   job.get("location", "N/A"),
                    "experience": job.get("experience", "N/A"),
                    "level":      job.get("level", "N/A"),
                    "job_type":   job.get("job_type", "N/A"),
                    "url":        job.get("url", "#"),
                    "source":     job.get("source", "live"),
                    "posted":     job.get("posted", None),
                    "match":      85.0  # درجة عالية للوظائف الحديثة من live scraping
                })
            print(f"✅ Live scraping got {len(matches)} jobs")

    # ── 7. normalize الـ scores بين 60% و 99% ──
    # عشان الـ scores تبان منطقية للمستخدم
    if matches:
        max_score = max(m['match'] for m in matches)
        min_score = min(m['match'] for m in matches)

        for m in matches:
            if max_score != min_score:
                normalized = 60 + ((m['match'] - min_score) / (max_score - min_score)) * 39
                m['match'] = round(normalized, 1)
            else:
                m['match'] = 75.0

    matches.sort(key=lambda x: x['match'], reverse=True)
    return matches, skills_profile


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            pdf_bytes = f.read()

        matches, profile = match_cv_to_jobs(pdf_bytes)
        print(f"\n🎯 Top matches:")
        for m in matches:
            print(f"  {m['match']}% | {m['title']} @ {m['company']}")