# scraper/wuzzuf_scraper.py
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json, time, os
import urllib.parse

BASE_URL = "https://wuzzuf.net"

SEARCH_QUERIES = [
    # Tech
    "software engineer",
    "python developer",
    "data scientist",
    "machine learning",
    "frontend developer",
    "backend developer",
    "full stack developer",
    "mobile developer",
    "flutter developer",
    "react developer",
    "devops engineer",
    "cloud engineer",
    "cybersecurity",
    "ui ux designer",
    "product manager",
    "data analyst",
    "data engineer",
    "AI engineer",
    "QA engineer",
    "odoo developer",

    # Business
    "marketing",
    "digital marketing",
    "sales",
    "business development",
    "customer service",
    "account manager",
    "social media",
    "content creator",
    "SEO specialist",
    "graphic designer",

    # Finance & Admin
    "accountant",
    "financial analyst",
    "HR specialist",
    "recruitment",
    "operations",
    "supply chain",
    "logistics",
    "project manager",

    # Fresh Grads
    "fresh graduate",
    "internship",
    "training",
    "entry level",
]

def parse_jobs_from_html(html):
    soup = BeautifulSoup(html, 'lxml')
    jobs = []

    job_links = soup.find_all('a', href=lambda x: x and '/jobs/p/' in x) #seach by jobs to ensure we get the job cards

    for link in job_links: #get the parent card of the link to get the company and location
        card = link.find_parent('div', class_=lambda x: x and 'css-lptxge' in x)
        if not card:
            card = link.find_parent('div')

        job = {}
        job['title'] = link.get_text(strip=True)
        job['url'] = BASE_URL + link['href'] if link['href'].startswith('/') else link['href']

        company_tag = card.find('a', class_=lambda x: x and 'css-ipsyv7' in x)
        job['company'] = company_tag.get_text(strip=True).replace(' -', '').strip() if company_tag else "N/A"

        location_tag = card.find('span', class_=lambda x: x and 'css-16x61xq' in x)
        job['location'] = location_tag.get_text(strip=True) if location_tag else "N/A"

        time_tag = card.find('div', class_=lambda x: x and 'css-eg55jf' in x)
        job['posted'] = time_tag.get_text(strip=True) if time_tag else "N/A"

        jobs.append(job)

    return jobs


def scrape_by_category(query, num_pages=3): #scrape by category to ensure we get more jobs and not just the most recent ones, also we can specify the number of pages to scrape for each category
    """scrape بـ keyword معين"""
    encoded = urllib.parse.quote(query)
    all_jobs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for page_num in range(num_pages):
            url = f"{BASE_URL}/search/jobs/?q={encoded}&a=1&filters[country][0]=Egypt&start={page_num * 15}"
            print(f"  📄 [{query}] Page {page_num + 1}")

            page.goto(url)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(2000)

            jobs = parse_jobs_from_html(page.content())
            all_jobs.extend(jobs)
            time.sleep(1.5)

        browser.close()

    return all_jobs


if __name__ == "__main__":
    all_jobs = []
    seen_urls = set()

    for query in SEARCH_QUERIES:
        print(f"\n🔍 Scraping: '{query}'")
        jobs = scrape_by_category(query, num_pages=3) #scrape 3 pages for each query, each page has 15 jobs, 

        for job in jobs:
            if job['url'] not in seen_urls:
                seen_urls.add(job['url'])
                all_jobs.append(job)

        print(f"  ✅ Unique jobs so far: {len(all_jobs)}")
        time.sleep(2)

    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/wuzzuf_jobs.json", "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Total unique jobs: {len(all_jobs)}")