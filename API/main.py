# API/main.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pathlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from RAG.pipeline import ask_stream, ask_non_stream, hybrid_search
from fastapi import UploadFile, File
from RAG.cv_matcher import match_cv_to_jobs
from groq import Groq
from dotenv import load_dotenv
import json
import chromadb
from chromadb.config import Settings as ChromaSettings

# Import routers
from prediction.prediction_routes import router as prediction_router
from agents.agent_routes import router as agent_router
from notification_engine.notification_routes import router as notification_router

# n8n requests
import requests

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq for cv reading
groq_client = Groq(api_key=GROQ_API_KEY)

# Init FastAPI server
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Direct ChromaDB client (for accurate counts) ──
chroma_client = chromadb.PersistentClient(
    path="database/chroma_db",
    settings=ChromaSettings(anonymized_telemetry=False)
)
chroma_collection = chroma_client.get_or_create_collection(name="jobs")

# ── Static Files ──
app.mount("/src", StaticFiles(directory="src"), name="src")

# Include routers
app.include_router(prediction_router)
app.include_router(agent_router)
app.include_router(notification_router)


class SearchRequest(BaseModel):
    query: str
    sources: list = None


# ── Streaming search ──
@app.post("/search/stream")
async def search_stream(req: SearchRequest):
    return StreamingResponse(
        ask_stream(req.query, sources=req.sources),
        media_type="text/event-stream"
    )


# ── Recent Opportunities (reads directly from ChromaDB) ──
@app.get("/api/recent-opportunities")
async def get_recent_opportunities(limit: int = 8):
    """Get recently posted job opportunities directly from ChromaDB"""
    try:
        result = chroma_collection.get(
            limit=limit,
            include=["metadatas"]
        )
        opportunities = []
        for meta in result["metadatas"]:
            opportunities.append({
                "title":      meta.get("title", "N/A"),
                "company":    meta.get("company", "N/A"),
                "location":   meta.get("location", "N/A"),
                "experience": meta.get("experience", "N/A"),
                "level":      meta.get("level", "N/A"),
                "job_type":   meta.get("job_type", "N/A"),
                "posted":     meta.get("posted", "Recently"),
                "url":        meta.get("url", "#"),
                "source":     meta.get("source", "wuzzuf"),
                "match":      85
            })
        return {"opportunities": opportunities, "count": len(opportunities)}
    except Exception as e:
        return {"error": str(e), "opportunities": [], "count": 0}


# ── Dashboard Stats (reads directly from ChromaDB) ──
@app.get("/api/dashboard-stats")
async def get_dashboard_stats():
    """Get real dashboard statistics directly from ChromaDB"""
    try:
        result = chroma_collection.get(include=["metadatas"])
        all_meta = result["metadatas"]

        total = len(all_meta)

        source_counts = {}
        for m in all_meta:
            src = m.get("source", "wuzzuf")
            source_counts[src] = source_counts.get(src, 0) + 1

        return {
            "totalOpportunities": total,
            "newThisWeek": source_counts.get("linkedin", 0) + source_counts.get("wuzzuf", 0),
            "applicationsActive": 5,
            "interviewsScheduled": 1,
            "activeSources": len(source_counts),
            "profileScore": 82,
            "freshJobs": total,
            "sourceDistribution": [
                {"label": k, "value": v} for k, v in source_counts.items()
            ]
        }
    except Exception as e:
        return {"error": str(e)}


# ── CV Matching ──
@app.post("/cv-match")
async def cv_match(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    matches, skills_profile = await match_cv_to_jobs(pdf_bytes, n_results=5)

    if not matches:
        return {"error": "مش قادر أقرأ الـ CV"}

    jobs_text = "\n".join([
        f"- {m['title']} @ {m['company']} ({m['match']}% match)"
        for m in matches
    ])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""بناءً على الـ CV، دي أنسب الوظايف:
            {jobs_text}

            اشرح بالعربي ليه الوظايف دي مناسبة في 3 جمل بس."""
        }],
        max_tokens=300
    )

    return {
        "matches": matches,
        "summary": response.choices[0].message.content
    }


# ── Non-streaming search (for testing/debugging) ──
@app.post("/search/non-stream")
async def search_non_stream(req: SearchRequest):
    answer, jobs = await ask_non_stream(req.query, sources=req.sources)
    print(f"Jobs found: {len(jobs)}")
    print(f"First job: {jobs[0] if jobs else 'NONE'}")
    return {"answer": answer, "jobs": jobs}


# ── Page Routes ──
@app.get("/")
async def root():
    return FileResponse("src/pages/dashboard.html")

@app.get("/dashboard")
async def dashboard_page():
    return FileResponse("src/pages/dashboard.html")

@app.get("/search")
async def search_page():
    return FileResponse("src/pages/search.html")

@app.get("/agent")
async def agent_page():
    return FileResponse("src/pages/agent.html")

@app.get("/prediction")
async def prediction_page():
    return FileResponse("src/pages/prediction.html")

@app.get("/notifications")
async def notifications_page():
    return FileResponse("src/pages/notifications.html")


# ── n8n Integration ──
def trigger_n8n_webhook(job_details):
    webhook_url = "http://localhost:5678/webhook-test/job-alert"
    payload = {
        "title":   job_details.get("title", "Unknown Job"),
        "company": job_details.get("company", "Unknown Company"),
        "url":     job_details.get("url", "https://linkedin.com")
    }
    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            print("✅ Webhook triggered successfully!")
        else:
            print(f"❌ Failed to push to n8n {response.text}")
    except Exception as e:
        print(f"❌ Connection error to n8n: {e}")


@app.get("/api/test-n8n")
def test_n8n_integration():
    test_job = {
        "title":   "Senior Python Developer",
        "company": "Tech Valley",
        "url":     "https://wuzzuf.net/jobs/123"
    }
    trigger_n8n_webhook(test_job)
    return {"status": "Test sent to n8n!"}