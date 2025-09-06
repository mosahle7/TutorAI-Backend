import weaviate
import os
from collections import Counter
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from weaviate.util import generate_uuid5
from weaviate.classes.query import Filter
import tqdm
from .client import connect_with_retry
from .utils import extract_pdf_text, parse, chunking, build_doc_terms, is_pdf
import re

def get_latest_file(data_dir):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,f))] 
    if not files:
        return FileNotFoundError("No files found in the directory.")
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def initialize_client():
    client = connect_with_retry()
    print(client.is_ready())
    return client

def initialize_collection(client):
    data_dir = "/root/TutorAI/backend/app/data"
    chunks_dir = "/root/TutorAI/backend/app/chunks"
    terms_dir = "/root/TutorAI/backend/app/terms"

    file = get_latest_file(data_dir)
    
    base_name = os.path.splitext(os.path.basename(file))[0]
    base_name = re.sub(r"[^A-Za-z0-9_]+","_",base_name)

    chunks_save = os.path.join(chunks_dir, base_name+".txt")
    os.makedirs(chunks_dir, exist_ok=True)

    terms_save = os.path.join(terms_dir, base_name+".txt")
    os.makedirs(terms_dir, exist_ok=True)

    if is_pdf(file):
        file = extract_pdf_text(file, os.path.join(data_dir,base_name))

    # file = "/root/TutorAI/backend/app/data/networks"
    if not client.collections.exists(base_name):
        with open(file,"r") as f:
            text = f.read()

        sections = parse(text)
        chunk_objs = chunking(sections)
        terms = build_doc_terms(chunk_objs)
        
        with open(chunks_save, "w", encoding="utf-8") as f:
            for obj in chunk_objs:
                f.write(f"{obj['chunk_id']}\n{obj['section']}\n{obj['text']}\n\n")

        with open(terms_save, "w", encoding="utf-8") as f:
            for term in terms:
                f.write(f"{term}\n")
                
        collection = client.collections.create(
            name = base_name,

            vectorizer_config = Configure.Vectorizer.text2vec_transformers(
                vectorize_collection_name = False,
                inference_url = "http://localhost:8000/weaviate"
            ),

            properties = [
                Property(name='chunk_id',data_type=DataType.TEXT),
                Property(name="section", vectorize_property_name = True, data_type=DataType.TEXT),
                Property(name="text", vectorize_property_name = True, data_type=DataType.TEXT)
            ]
        )
        print(f"Collection: {base_name} created successfully!")
    else:
        collection = client.collections.get(base_name)
        with open(terms_save,"r",encoding="utf-8") as f:
            terms = [line.strip() for line in f if line.strip()]
        print("Using existing collection: ",base_name)


    collection = client.collections.get(base_name)

    if len(collection) == 0:
        with collection.batch.fixed_size(batch_size=200,concurrent_requests=1) as batch:
            for obj in tqdm.tqdm(chunk_objs):
                    batch.add_object(
                        properties = obj,
                        uuid=generate_uuid5(obj["chunk_id"])
                    )

    print("Length of collection: ",len(collection))

    return collection, terms

