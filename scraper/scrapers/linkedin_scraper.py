# scraper/scrapers/linkedin_scraper.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json, time, os, urllib.parse

BASE_URL = "https://www.linkedin.com"

SEARCH_QUERIES = [
    "software engineer Egypt",
    "python developer Egypt",
    "data scientist Egypt",
    "frontend developer Egypt",
    "backend developer Egypt",
    "machine learning Egypt",
    "data analyst Egypt",
    "UI UX designer Egypt",
    "DevOps engineer Egypt",
    "fresh graduate Egypt",
]

def parse_jobs_from_html(html):
    soup = BeautifulSoup(html, 'lxml')
    jobs = []

    job_cards = soup.find_all('div', class_=lambda x: x and 'job-search-card' in x)

    for card in job_cards:
        try:
            title_tag = card.find('h3', class_=lambda x: x and 'base-search-card__title' in x)
            company_tag = card.find('h4', class_=lambda x: x and 'base-search-card__subtitle' in x)
            location_tag = card.find('span', class_=lambda x: x and 'job-search-card__location' in x)
            link_tag = card.find('a', class_=lambda x: x and 'base-card__full-link' in x)
            time_tag = card.find('time')

            if not title_tag or not link_tag:
                continue

            jobs.append({
                "title":       title_tag.get_text(strip=True),
                "company":     company_tag.get_text(strip=True) if company_tag else "N/A",
                "location":    location_tag.get_text(strip=True) if location_tag else "Egypt",
                "url":         link_tag['href'].split('?')[0],  # clean URL
                "posted":      time_tag.get('datetime', 'N/A') if time_tag else "N/A",
                "job_type":    "N/A",
                "experience":  "N/A",
                "level":       "N/A",
                "description": "N/A",
                "requirements":"N/A",
                "source":      "linkedin",
            })
        except Exception as e:
            continue

    return jobs


def scrape_linkedin(query, num_pages=3):
    encoded = urllib.parse.quote(query)
    all_jobs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # set a realistic user agent
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        for page_num in range(num_pages):
            start = page_num * 25
            url = f"https://www.linkedin.com/jobs/search/?keywords={encoded}&location=Egypt&start={start}"
            print(f"  📄 [{query}] Page {page_num + 1}")

            try:
                page.goto(url, timeout=30000)
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(3000)

                # scroll to load more jobs
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(2000)

                jobs = parse_jobs_from_html(page.content())
                print(f"     → Found {len(jobs)} jobs")
                all_jobs.extend(jobs)

            except Exception as e:
                print(f"     ❌ Error on page {page_num + 1}: {e}")

            time.sleep(2)

        browser.close()

    return all_jobs


if __name__ == "__main__":
    all_jobs = []
    seen_urls = set()

    for query in SEARCH_QUERIES:
        print(f"\n🔍 Scraping LinkedIn: '{query}'")
        jobs = scrape_linkedin(query, num_pages=2)

        for job in jobs:
            if job['url'] not in seen_urls:
                seen_urls.add(job['url'])
                all_jobs.append(job)

        print(f"  ✅ Unique jobs so far: {len(all_jobs)}")
        time.sleep(3)  # be polite to LinkedIn

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/linkedin_full.json", "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Total LinkedIn jobs: {len(all_jobs)}")
    print("📁 Saved to data/processed/linkedin_full.json")