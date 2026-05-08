# database/migrate_to_pinecone.py
"""
Migration script: Chroma DB → Pinecone Vector DB
Transfers all jobs, embeddings, and metadata to Pinecone cloud
"""

import os
import json
from dotenv import load_dotenv
import chromadb
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import time

load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Get from https://app.pinecone.io
PINECONE_INDEX_NAME = "jobs-rag-egypt"
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDING_DIMENSION = 384  # Output dimension of MiniLM model
BATCH_SIZE = 100  # Pinecone batch upload size

# Initialize
from chromadb.config import Settings
pc = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
chroma_client = chromadb.PersistentClient(
    path="database/chroma_db",
    settings=Settings(anonymized_telemetry=False)
)


def create_pinecone_index():
    """Create Pinecone index if it doesn't exist"""
    print("🔧 Checking Pinecone index...")
    
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]
    
    if PINECONE_INDEX_NAME in index_names:
        print(f"✅ Index '{PINECONE_INDEX_NAME}' already exists")
        return
    
    print(f"📝 Creating index '{PINECONE_INDEX_NAME}'...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Change region as needed
        )
    )
    
    # Wait for index to be ready
    print("⏳ Waiting for index to be ready...")
    while not pc.describe_index(PINECONE_INDEX_NAME).status.ready:
        time.sleep(1)
    
    print("✅ Index ready!")


def migrate_jobs_to_pinecone():
    """Extract jobs from Chroma and upload to Pinecone"""
    print("\n📤 Starting migration...")
    
    # Get index
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Get Chroma collection
    collection = chroma_client.get_collection("jobs")
    
    # Fetch all data from Chroma
    all_data = collection.get(include=["documents", "metadatas", "embeddings"])
    
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    embeddings = all_data.get("embeddings", None)
    ids = all_data["ids"]
    
    total_jobs = len(documents)
    print(f"📊 Found {total_jobs} jobs in Chroma DB")
    
    # If embeddings weren't returned, regenerate them
    if embeddings is None or len(embeddings) == 0:
        print("🔄 Regenerating embeddings (not cached in Chroma)...")
        embeddings = model.encode(documents, show_progress_bar=True)
        # Convert to regular Python lists
        embeddings = [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
    else:
        # Convert to list if it's numpy array
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()
        else:
            # Handle case where embeddings is already a list
            new_embeddings = []
            for emb in embeddings:
                if hasattr(emb, 'tolist'):
                    new_embeddings.append(emb.tolist())
                else:
                    new_embeddings.append(list(emb))
            embeddings = new_embeddings
    
    # Prepare vectors for Pinecone
    vectors_to_upsert = []
    batch_count = 0
    
    for i in range(total_jobs):
        doc_id = ids[i]
        embedding = embeddings[i]
        metadata = metadatas[i]
        
        # Pinecone expects: (id, values, metadata)
        vector = (
            doc_id,
            embedding,
            metadata  # All metadata stored here
        )
        vectors_to_upsert.append(vector)
        
        # Upload in batches
        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total_jobs:
            batch_count += 1
            actual_batch_size = len(vectors_to_upsert)
            print(f"📤 Uploading batch {batch_count} ({actual_batch_size} items)...")
            
            if actual_batch_size > 0:
                index.upsert(vectors=vectors_to_upsert)
            
            vectors_to_upsert = []
            
            # Small delay to avoid rate limiting
            if (i + 1) < total_jobs:
                time.sleep(0.5)
    
    print(f"\n✅ Migration complete! {total_jobs} jobs uploaded to Pinecone")
    
    # Verify
    stats = index.describe_index_stats()
    print(f"📊 Pinecone Index Stats:")
    print(f"   Total vectors: {stats.total_vector_count}")
    print(f"   Dimension: {stats.dimension}")


def save_migration_info():
    """Save migration info for reference"""
    info = {
        "migration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Chroma DB (local)",
        "destination": f"Pinecone ({PINECONE_INDEX_NAME})",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "api_endpoint": f"https://api.pinecone.io",
        "setup_instructions": {
            "1_get_api_key": "Visit https://app.pinecone.io to get your API key",
            "2_set_env": "Add PINECONE_API_KEY to your .env file",
            "3_update_retriever": "Import and use PineconeRetriever instead of ChromaDB"
        }
    }
    
    with open("database/migration_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print("\n📋 Migration info saved to database/migration_info.json")


if __name__ == "__main__":
    try:
        if not PINECONE_API_KEY:
            print("❌ PINECONE_API_KEY not found in .env")
            print("📝 Follow these steps:")
            print("   1. Go to https://app.pinecone.io")
            print("   2. Create a free account (if needed)")
            print("   3. Copy your API key")
            print("   4. Add to .env: PINECONE_API_KEY=your_key_here")
            exit(1)
        
        create_pinecone_index()
        migrate_jobs_to_pinecone()
        save_migration_info()
        
        print("\n✨ All done! Your data is now in Pinecone cloud ✨")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
