from .ingestion import initialize_client, initialize_collection

client = initialize_client()
print("Weaviate initialized")

collection, terms = initialize_collection(client)

