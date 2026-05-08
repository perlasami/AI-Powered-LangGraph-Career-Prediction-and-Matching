# database/test_migration.py
"""
Test and verify Pinecone migration
Compares Chroma vs Pinecone results to ensure data integrity
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import json
from datetime import datetime

load_dotenv()

# Import both retrievers for comparison
from database.pinecone_retriever import simple_vector_search as pinecone_search
from database.hybrid_retriever import hybrid_search as chroma_search

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def test_chroma_data():
    """Verify Chroma DB has data"""
    print("\n🔍 Testing Chroma DB...")
    try:
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path="database/chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection("jobs")
        count = collection.count()
        print(f"   ✅ Chroma DB has {count} jobs")
        return count
    except Exception as e:
        print(f"   ❌ Chroma DB error: {e}")
        return 0


def test_pinecone_connection():
    """Test Pinecone connectivity"""
    print("\n🔗 Testing Pinecone connection...")
    try:
        from pinecone import Pinecone
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("   ❌ PINECONE_API_KEY not set")
            return False
        
        pc = Pinecone(api_key=api_key)
        index_name = os.getenv("PINECONE_INDEX_NAME", "jobs-rag-egypt")
        index = pc.Index(index_name)
        
        stats = index.describe_index_stats()
        count = stats.total_vector_count
        print(f"   ✅ Pinecone connected - {count} vectors in index")
        return True, count
    except Exception as e:
        print(f"   ❌ Pinecone error: {e}")
        return False, 0


def compare_search_results(test_queries):
    """Compare Chroma vs Pinecone results"""
    print("\n📊 Comparing search results...\n")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_queries": test_queries,
        "comparisons": []
    }
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 50)
        
        try:
            # Chroma results
            chroma_results = chroma_search(query, n_results=5)
            chroma_titles = [r['job'].get('title', 'N/A') for r in chroma_results]
            
            # Pinecone results
            pinecone_results = pinecone_search(query, n_results=5)
            pinecone_titles = [r['job'].get('title', 'N/A') for r in pinecone_results]
            
            # Compare
            common = set(chroma_titles) & set(pinecone_titles)
            similarity = len(common) / 5 if chroma_titles else 0
            
            print(f"Chroma (top 5):   {chroma_titles[:3]}...")
            print(f"Pinecone (top 5): {pinecone_titles[:3]}...")
            print(f"Match rate: {similarity * 100:.0f}%")
            print()
            
            results["comparisons"].append({
                "query": query,
                "chroma_results": chroma_titles,
                "pinecone_results": pinecone_titles,
                "match_rate": similarity
            })
        
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    return results


def check_metadata_integrity():
    """Verify metadata transferred correctly"""
    print("\n🔐 Checking metadata integrity...\n")
    
    try:
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "jobs-rag-egypt"))
        
        # Check a sample vector
        stats = index.describe_index_stats()
        print(f"   ✅ Index dimension: {stats.dimension} (expected: 384)")
        print(f"   ✅ Total vectors: {stats.total_vector_count}")
        
        # Query with limit to check metadata
        test_vector = model.encode("test").tolist()
        results = index.query(vector=test_vector, top_k=3, include_metadata=True)
        
        if results.matches:
            sample = results.matches[0]
            metadata = sample.metadata
            
            required_fields = ['title', 'company', 'location', 'url', 'source']
            missing = [f for f in required_fields if f not in metadata]
            
            if not missing:
                print(f"   ✅ All metadata fields present:")
                for k, v in list(metadata.items())[:5]:
                    print(f"      • {k}: {str(v)[:50]}")
            else:
                print(f"   ⚠️  Missing fields: {missing}")
    
    except Exception as e:
        print(f"   ❌ Metadata check failed: {e}")


def run_full_test():
    """Run complete migration test"""
    print("\n" + "=" * 60)
    print("     MIGRATION TEST SUITE")
    print("=" * 60)
    
    # 1. Test Chroma
    chroma_count = test_chroma_data()
    
    # 2. Test Pinecone
    pinecone_ok, pinecone_count = test_pinecone_connection()
    
    if not pinecone_ok:
        print("\n❌ Pinecone not configured. Run migration first:")
        print("   python database/migrate_to_pinecone.py")
        return
    
    # 3. Verify counts match
    print(f"\n✅ Vector counts:")
    print(f"   Chroma: {chroma_count}")
    print(f"   Pinecone: {pinecone_count}")
    
    if chroma_count == pinecone_count:
        print("   ✅ Counts match!")
    else:
        print(f"   ⚠️  Count mismatch (difference: {abs(chroma_count - pinecone_count)})")
    
    # 4. Check metadata
    check_metadata_integrity()
    
    # 5. Compare search results
    test_queries = [
        "python developer",
        "data scientist cairo",
        "senior engineer"
    ]
    
    comparison_results = compare_search_results(test_queries)
    
    # 6. Save test results
    with open("database/migration_test_results.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print("=" * 60)
    print("✅ Test complete! Results saved to migration_test_results.json")
    print("=" * 60)


if __name__ == "__main__":
    run_full_test()
