from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(":memory:")  # Use in-memory for demo; replace with URL for persistent

def init_qdrant(collection_name):
    if collection_name not in client.get_collections().collections:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )

def insert_vectors(collection_name, vectors, payloads):
    points = [
        PointStruct(id=i, vector=vec, payload=payload)
        for i, (vec, payload) in enumerate(zip(vectors, payloads))
    ]
    client.upsert(collection_name=collection_name, points=points)

def search_vectors(collection_name, vector, top_k=5):
    return client.search(collection_name=collection_name, query_vector=vector, limit=top_k, with_vectors=True, with_payload=True)
