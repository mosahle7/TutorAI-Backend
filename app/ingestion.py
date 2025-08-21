import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from weaviate.util import generate_uuid5
from weaviate.classes.query import Filter
import tqdm
from .client import connect_with_retry
from .utils import parse, chunking

def initialize_weaviate():
    client = connect_with_retry()
    print(client.is_ready())

    file = "/root/TutorAI/backend/app/data/networks"

    with open(file,"r") as f:
        text = f.read()


    sections = parse(text)

    chunk_objs = chunking(sections)


    if not client.collections.exists('networks'):
        collection = client.collections.create(
            name = 'networks',

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
        print("Collection created successfully!")
    else:
        collection = client.collections.get('networks')
        print("Using existing collection.")


    collection = client.collections.get('networks')

    if len(collection) == 0:
        with collection.batch.fixed_size(batch_size=200,concurrent_requests=1) as batch:
            for obj in tqdm.tqdm(chunk_objs):
                    batch.add_object(
                        properties = obj,
                        uuid=generate_uuid5(obj["chunk_id"])
                    )

    print("Length of collection: ",len(collection))

    return client, collection

