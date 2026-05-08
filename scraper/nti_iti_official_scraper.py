def nti_iti_details(job):
    return {
        "title": job.get("title"),
        "company": job.get("company"),
        "location": job.get("location"),
        "description": job.get("description"),
        "url": job.get("url"),
        "type": "training",
        "level": "entry"
    }