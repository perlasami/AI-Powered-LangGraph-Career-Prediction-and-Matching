from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from groq import Groq
import os
from dotenv import load_dotenv
import json
from difflib import SequenceMatcher
import datetime

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════
# Freshness & Status Filter Functions
# ═══════════════════════════════════════════════════════════════
def is_job_open_and_fresh(job: Dict) -> bool:
    """
    تحقق من أن الوظيفة:
    1. مفتوحة (status = 'open')
    2. حديثة (posted خلال آخر 14 يوم)
    """
    # فحص الحالة
    status = job.get('status', '').lower()
    if status and status != 'open':
        return False
    
    # فحص تاريخ النشر
    posted_date_str = job.get('posted', '')
    if posted_date_str:
        try:
            # جرب صيغة مختلفة من التواريخ
            for date_format in ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]:
                try:
                    posted_date = datetime.datetime.strptime(posted_date_str, date_format).date()
                    today = datetime.date.today()
                    delta_days = (today - posted_date).days
                    
                    # خلال آخر 14 يوم فقط
                    if delta_days < 0:  # تاريخ مستقبلي، اقبلها
                        return True
                    if delta_days <= 14:
                        return True
                    else:
                        return False  # قديمة جداً
                except ValueError:
                    continue
            # إذا فشل parsing في جميع الصيغ، اعتبر الوظيفة قديمة
            return False
        except Exception:
            return True  # في حالة الخطأ، اعتبرها fresh
    
    return True  # بدون تاريخ = نفترض أنها حديثة

def filter_fresh_open_jobs(jobs: List[Dict]) -> List[Dict]:
    """فلترة النتائج بحيث تكون الوظائف مفتوحة وحديثة فقط"""
    filtered = []
    for job in jobs:
        if is_job_open_and_fresh(job):
            filtered.append(job)
    return filtered

# ═══════════════════════════════════════════════════════════════
# Load all job data at module startup
# ═══════════════════════════════════════════════════════════════
_jobs_cache = {}

def _load_job_data():
    """Load all job data from JSON files"""
    global _jobs_cache
    try:
        # Load LinkedIn data
        linkedin_path = "data/processed/linkedin_full.json"
        if os.path.exists(linkedin_path):
            with open(linkedin_path, "r", encoding="utf-8") as f:
                linkedin_jobs = json.load(f)
                _jobs_cache["linkedin"] = linkedin_jobs
                print(f"✅ Loaded {len(linkedin_jobs)} LinkedIn jobs")
        else:
            _jobs_cache["linkedin"] = []
            print("ℹ️  LinkedIn data not found, skipping")
    except Exception as e:
        print(f"⚠️  Failed to load LinkedIn data: {e}")
        _jobs_cache["linkedin"] = []
    
    try:
        # Load Wuzzuf data
        wuzzuf_path = "data/processed/wuzzuf_full.json"
        if os.path.exists(wuzzuf_path):
            with open(wuzzuf_path, "r", encoding="utf-8") as f:
                wuzzuf_jobs = json.load(f)
                _jobs_cache["wuzzuf"] = wuzzuf_jobs
                print(f"✅ Loaded {len(wuzzuf_jobs)} Wuzzuf jobs")
        else:
            _jobs_cache["wuzzuf"] = []
            print("⚠️  Wuzzuf data not found")
    except Exception as e:
        print(f"⚠️  Failed to load Wuzzuf data: {e}")
        _jobs_cache["wuzzuf"] = []

# Load data at module import
_load_job_data()

def _search_jobs_by_keywords(jobs: List[Dict], query: str, limit: int = 10) -> List[Dict]:
    """
    Search through jobs using keyword matching and similarity scoring.
    Returns jobs ranked by relevance to the query.
    ✅ تطبق فلترة البيانات الحديثة والمفتوحة
    """
    # فلترة أولاً: اترك الوظائف الحديثة والمفتوحة فقط
    fresh_jobs = filter_fresh_open_jobs(jobs)
    
    if not fresh_jobs:
        print(f"  ⚠️  No fresh/open jobs available, returning empty results")
        return []
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_jobs = []
    
    for job in fresh_jobs:
        title = job.get("title", "").lower()
        description = job.get("description", "").lower()
        company = job.get("company", "").lower()
        requirements = job.get("requirements", "").lower()
        
        # Calculate relevance score
        score = 0
        
        # Title matches are most important
        if any(word in title for word in query_words):
            score += 3
        
        # Company matches
        if any(word in company for word in query_words):
            score += 2
        
        # Description/Requirements matches
        if any(word in description for word in query_words):
            score += 1
        if any(word in requirements for word in query_words):
            score += 1
        
        # Similarity scoring
        title_similarity = SequenceMatcher(None, query_lower, title).ratio()
        description_similarity = SequenceMatcher(None, query_lower, description[:100]).ratio()
        
        score += (title_similarity * 2) + (description_similarity * 0.5)
        
        if score > 0:
            scored_jobs.append({
                "job": job,
                "score": score
            })
    
    # Sort by score and return top results
    scored_jobs.sort(key=lambda x: x["score"], reverse=True)
    return scored_jobs[:limit]

# State Definition
class AgentState(TypedDict):
    query: str
    opportunity_type: str
    filters: Optional[Dict]
    user_context: Optional[Dict]
    search_sources: List[str]
    results: List[Dict[str, Any]]
    final_response: str
    sources_used: List[str]

# 1. Router Node: decides which sources to query
def router_node(state: AgentState) -> AgentState:
    """
    LLM-based router that analyzes the query and determines relevant sources.
    Uses Groq LLM to intelligently route queries to appropriate data sources.
    """
    query = state["query"]
    print(f"🧭 Routing query: '{query}'")
    
    prompt = f"""
Given the user query: "{query}", decide which data sources are most relevant to search.
Possible sources: linkedin, wuzzuf, iti_nti, companies, facebook, volunteering, learning.

Rules:
- For general software/internship/job roles, prioritize 'linkedin' and 'wuzzuf' (Egypt's largest job site).
- If the user explicitly mentions Wuzzuf, choose 'wuzzuf'.
- If the user explicitly mentions Facebook or Facebook groups, choose 'facebook'.
- If the user explicitly mentions ITI, NTI, courses, academies, choose 'iti_nti'.
- If the user mentions working at a specific company officially or requests their official website, choose 'companies'.
- If the user mentions volunteering, choose 'volunteering'.
- If the user mentions learning, courses, or training, choose 'learning'.
- Default to 'linkedin,wuzzuf' for general job searches.

Output ONLY a comma-separated list of EXACT source names. 
Example output: linkedin,wuzzuf
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a routing agent. Output ONLY the comma-separated source names."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        sources_str = response.choices[0].message.content.lower().strip()
        
        # Parse the response
        selected_sources = []
        for s in ["linkedin", "wuzzuf", "companies", "iti_nti", "facebook", "volunteering", "learning"]:
            if s in sources_str:
                selected_sources.append(s)
        
        if not selected_sources:
            selected_sources = ["linkedin", "wuzzuf"]  # fallback to both main sources
        
        state["search_sources"] = selected_sources
        print(f"  ✓ Routing to sources: {', '.join(selected_sources)}")
        return state
    except Exception as e:
        print(f"⚠️  Router error: {e}, defaulting to linkedin,wuzzuf")
        state["search_sources"] = ["linkedin", "wuzzuf"]
        return state


# 2. Retrieval Functions (synchronous)
def search_linkedin(query: str, n_results=5) -> List[Dict]:
    """
    Search LinkedIn jobs using Pinecone vector DB with hybrid search.
    Includes automatic freshness filtering.
    """
    try:
        from database.pinecone_retriever import hybrid_search
        results = hybrid_search(query, n_results=n_results)
        print(f"  📊 LinkedIn: Found {len(results)} fresh jobs")
        return results
    except Exception as e:
        print(f"  ⚠️  LinkedIn search error: {e}")
        return []


def search_wuzzuf(query: str, n_results=5) -> List[Dict]:
    """
    Search Wuzzuf job data using keyword matching and relevance scoring.
    Returns real job data from the project database.
    """
    wuzzuf_jobs = _jobs_cache.get("wuzzuf", [])
    if not wuzzuf_jobs:
        print(f"  📱 Wuzzuf: No data available")
        return []
    
    results = _search_jobs_by_keywords(wuzzuf_jobs, query, limit=n_results)
    print(f"  📱 Wuzzuf: Found {len(results)} matching jobs")
    return results


def search_facebook(query: str) -> List[Dict]:
    """
    Search Wuzzuf jobs (treated as second source for better coverage).
    Using actual job data instead of mock.
    """
    print(f"  📘 Facebook: Searching (Wuzzuf fallback)...")
    # Use Wuzzuf as fallback for broader coverage
    return search_wuzzuf(query, n_results=3)


def search_iti_nti(query: str) -> List[Dict]:
    """
    Search for training and educational programs.
    Currently filtering jobs by keywords like 'training', 'course', 'certification' etc.
    """
    print(f"  🎓 ITI/NTI: Searching programs...")
    all_jobs = _jobs_cache.get("linkedin", []) + _jobs_cache.get("wuzzuf", [])
    
    training_keywords = ["training", "course", "certificate", "certification", "bootcamp", "academy", "program"]
    training_jobs = [j for j in all_jobs if any(kw in j.get("title", "").lower() + j.get("description", "").lower() for kw in training_keywords)]
    
    results = _search_jobs_by_keywords(training_jobs, query, limit=5)
    print(f"  🎓 ITI/NTI: Found {len(results)} training programs")
    return results


def search_companies(query: str) -> List[Dict]:
    """
    Search company career pages using real job data.
    Looks for jobs from major Egyptian/International companies.
    """
    print(f"  🏢 Companies: Searching career pages...")
    linkedin_jobs = _jobs_cache.get("linkedin", [])
    
    results = _search_jobs_by_keywords(linkedin_jobs, query, limit=5)
    print(f"  🏢 Companies: Found {len(results)} company jobs")
    return results


# 3. Retriever Node
def retriever_node(state: AgentState) -> AgentState:
    """
    Executes searches across all selected sources.
    Aggregates results from multiple sources with real job data.
    """
    query = state["query"]
    sources = state["search_sources"]
    all_results = []
    used_sources = []
    
    print(f"\n🔍 Searching {len(sources)} source(s)...")
    
    # Execute searches based on selected sources
    if "linkedin" in sources:
        results = search_linkedin(query, n_results=5)
        all_results.extend(results)
        if results:
            used_sources.append("linkedin")
    
    if "wuzzuf" in sources or "facebook" in sources:
        # Treat wuzzuf as main source or facebook fallback
        results = search_wuzzuf(query, n_results=5)
        all_results.extend(results)
        if results:
            used_sources.append("wuzzuf")
    
    if "facebook" in sources and "wuzzuf" not in sources:
        results = search_facebook(query)
        all_results.extend(results)
        if results:
            used_sources.append("facebook")
    
    if "iti_nti" in sources:
        results = search_iti_nti(query)
        all_results.extend(results)
        if results:
            used_sources.append("iti_nti")
    
    if "companies" in sources:
        results = search_companies(query)
        all_results.extend(results)
        if results:
            used_sources.append("companies")
    
    if "volunteering" in sources:
        print(f"  🤝 Volunteering: [Not yet implemented]")
    
    if "learning" in sources:
        print(f"  📚 Learning: [Not yet implemented]")
    
    state["results"] = all_results
    state["sources_used"] = used_sources
    print(f"  ✓ Total results from {len(used_sources)} source(s): {len(all_results)} opportunities\n")
    return state


# 4. Aggregator/Ranker Node
def aggregator_node(state: AgentState) -> AgentState:
    """
    Formats and explains the search results using LLM.
    Generates a comprehensive summary of why results match the user query.
    """
    query = state["query"]
    results = state.get("results", [])
    
    if not results:
        state["final_response"] = "لم أستطع العثور على أي فرص مناسبة. حاول تعديل بحثك أو المحاولة لاحقاً."
        return state
    
    print(f"📝 Formatting {len(results)} results...")
    
    # Format results for LLM context
    text_context = "Opportunities found:\n\n"
    for i, r in enumerate(results[:10], 1):
        job = r.get('job', {})
        source = job.get('source', 'unknown')
        text_context += f"{i}. {job.get('title', 'N/A')}\n"
        text_context += f"   Company: {job.get('company', 'N/A')}\n"
        text_context += f"   Location: {job.get('location', 'N/A')}\n"
        text_context += f"   Source: {source}\n"
        text_context += f"   URL: {job.get('url', 'N/A')}\n\n"
    
    prompt = f"""You are an intelligent career opportunity recommendation agent.

User Query: "{query}"

Here are opportunities found across multiple sources:
{text_context}

Write a brief, encouraging summary (in Arabic) explaining why these opportunities match the user's request.
Focus on relevance and potential. Be concise (2-3 sentences max).
Do NOT list the opportunities yourself - just explain the relevance."""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful career advisor speaking Arabic."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        state["final_response"] = response.choices[0].message.content
        print(f"✓ Aggregation complete")
        return state
    except Exception as e:
        print(f"⚠️  Aggregator error: {e}")
        state["final_response"] = f"تم العثور على {len(results)} فرصة متعلقة بـ: {query}\nتفاصيل الفرص موضحة أعلاه."
        return state


# ═══════════════════════════════════════════════════════════════
# 5. Build and Compile the LangGraph
# ═══════════════════════════════════════════════════════════════
def build_graph():
    """
    Constructs the LangGraph state machine.
    
    Flow:
    START → Router → Retriever → Aggregator → END
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("aggregator", aggregator_node)
    
    # Define flow
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "aggregator")
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()


# Compile the graph at module load time
dist_search_app = build_graph()

