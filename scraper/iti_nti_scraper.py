# scraper/nti_iti_official_scraper.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import requests
from bs4 import BeautifulSoup
import json, time, os, urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ============================================================
# البرامج الثابتة لـ NTI — مأخوذة من الموقع الرسمي مباشرة
# ============================================================
NTI_PROGRAMS = [
    {
        "title":       "Egyptian Talents Academy (ETA) — Huawei",
        "url":         "https://nti.sci.eg/eta/",
        "description": "منحة كاملة — تدريب في AI, Cloud, 5G, Security, Datacom. للخريجين وطلاب السنة الأخيرة.",
        "contact":     "ETA@nti.sci.eg",
    },
    {
        "title":       "Digital Egypt Youth (DEY) — Evening Scholarship",
        "url":         "https://www.nti.sci.eg/dey/specialized_programs.html",
        "description": "منحة مسائية — 60 إلى 120 ساعة تقنية وsoft skills. أونلاين أو حضوري في كل المحافظات.",
        "contact":     "deyevening@nti.sci.eg",
    },
    {
        "title":       "Summer Training — NTI & ITIDA",
        "url":         "https://www.nti.sci.eg/summertraining/summermodules.php",
        "description": "تدريب صيفي 4 أسابيع — 120 ساعة (90 تقنية + 30 soft skills). لطلاب سنة 1 و2 و3 هندسة وحاسبات.",
        "contact":     "training@nti.sci.eg",
    },
    {
        "title":       "Private Summer Training",
        "url":         "https://www.nti.sci.eg/pst/",
        "description": "تدريب صيفي خاص مدفوع — يتطلب كود من الفريق المالي للتسجيل.",
        "contact":     "training@nti.sci.eg",
    },
    {
        "title":       "Creativa Innovation Hubs — Free Work Initiative",
        "url":         "https://nti.sci.eg/",
        "description": "ورش عمل ومبادرات في أكتر من 24 مركز في كل المحافظات. نسبة توظيف 73%+ كـ freelance.",
        "contact":     "training@nti.sci.eg",
    },
]

# ============================================================
# البرامج الثابتة لـ ITI
# ============================================================
ITI_PROGRAMS = [
    {
        "title":       "ITI 9-Month Professional Diploma",
        "url":         "https://iti.gov.eg/home",
        "description": "دبلومة مكثفة 9 أشهر في تخصصات مختلفة — البرمجة، الشبكات، الجرافيك، وغيرها.",
        "contact":     "info@iti.gov.eg",
    },
    {
        "title":       "ITI Summer Training",
        "url":         "https://iti.gov.eg/home",
        "description": "تدريب صيفي مكثف للطلاب الجامعيين في مجال تكنولوجيا المعلومات.",
        "contact":     "info@iti.gov.eg",
    },
    {
        "title":       "ITI Internship Program",
        "url":         "https://iti.gov.eg/home",
        "description": "برنامج تدريب داخلي للخريجين الجدد في مجال IT.",
        "contact":     "info@iti.gov.eg",
    },
]

# ============================================================
# بنحاول نسكرايب الصفحة — لو فشل بنرجع البيانات الثابتة
# ============================================================
def try_scrape_page(url, source_name):
    """بيحاول يسكرايب الصفحة ويجيب أي بيانات إضافية"""
    try:
        res = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        if res.status_code != 200:
            return None

        soup = BeautifulSoup(res.text, "lxml")

        # بندور على أي كروت أو sections
        cards = soup.select(
            "div.card, div.program, article, "
            "div.col-md-4, div.col-lg-4, div.track, "
            "div.item, li.item"
        )

        extra_jobs = []
        for card in cards:
            title_tag = card.select_one("h2, h3, h4, .title")
            link_tag  = card.select_one("a[href]")
            desc_tag  = card.select_one("p, .description")

            if not title_tag:
                continue

            href = link_tag["href"] if link_tag else url
            if href.startswith("/"):
                href = "https://nti.sci.eg" + href

            extra_jobs.append({
                "title":       title_tag.get_text(strip=True),
                "company":     source_name,
                "location":    "Egypt",
                "url":         href,
                "posted":      "N/A",
                "job_type":    "Training / Internship",
                "experience":  "N/A",
                "level":       "Entry Level",
                "description": desc_tag.get_text(strip=True) if desc_tag else "N/A",
                "source":      source_name.lower().replace(" ", "_"),
            })

        return extra_jobs if extra_jobs else None

    except Exception:
        return None


def build_jobs_from_programs(programs, source_name):
    """يحوّل الـ programs list لـ jobs list"""
    jobs = []
    for p in programs:
        print(f"  🔗 [{source_name}] {p['title']}")

        # نحاول نسكرايب الصفحة أول
        scraped = try_scrape_page(p["url"], source_name)
        if scraped:
            jobs.extend(scraped)
            print(f"      → سكرايب الصفحة ورجع {len(scraped)} برنامج")
        else:
            # لو فشل، نضيف البيانات الثابتة
            jobs.append({
                "title":       p["title"],
                "company":     source_name,
                "location":    "Egypt - All Governorates",
                "url":         p["url"],
                "posted":      "N/A",
                "job_type":    "Training / Internship",
                "experience":  "N/A",
                "level":       "Entry Level",
                "description": p["description"],
                "source":      source_name.lower(),
            })
            print(f"      → استخدمنا البيانات الثابتة")

        time.sleep(1)

    return jobs


def scrape_nti():
    print("\n📡 [NTI] بنجيب برامج NTI...")
    return build_jobs_from_programs(NTI_PROGRAMS, "NTI")


def scrape_iti():
    print("\n🏫 [ITI] بنجيب برامج ITI...")
    return build_jobs_from_programs(ITI_PROGRAMS, "ITI")


def scrape_all():
    nti_jobs = scrape_nti()
    iti_jobs = scrape_iti()
    all_jobs = nti_jobs + iti_jobs

    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/nti_iti_official.json", "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 المجموع: {len(all_jobs)} برنامج")
    print(f"   NTI: {len(nti_jobs)} | ITI: {len(iti_jobs)}")
    print("📁 اتحفظ في data/raw/nti_iti_official.json")

    return all_jobs


if __name__ == "__main__":
    scrape_all()