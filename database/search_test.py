# database/search_test.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client = chromadb.PersistentClient(
    path="database/chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection("jobs")

def search_jobs(query, n_results=3):
    print(f"\n🔍 Query: '{query}'")
    print("=" * 50)

    # حول الـ query لـ vector
    query_vector = model.encode(query).tolist()

    # دور في الـ DB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )

    for i, metadata in enumerate(results['metadatas'][0]):
        print(f"\n#{i+1}")
        print(f"  Title:      {metadata['title']}")
        print(f"  Company:    {metadata['company']}")
        print(f"  Location:   {metadata['location']}")
        print(f"  Experience: {metadata['experience']}")
        print(f"  URL:        {metadata['url']}")


# ── جرب queries مختلفة ──
search_jobs("customer service english")
search_jobs("software engineer python cairo")
search_jobs("fresh grad entry level")