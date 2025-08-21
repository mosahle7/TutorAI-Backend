import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter, Rerank
from weaviate.util import generate_uuid5
import time

def connect_with_retry(max_retries=5, delay=2):
    for attempt in range(max_retries):
        try:
            client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                grpc_port=8081,
                skip_init_checks=True,
                additional_config=weaviate.classes.init.AdditionalConfig(
                    timeout=weaviate.classes.init.Timeout(init=30, query=60, insert=120)
                )
            )
            
            # Test the connection
            if client.is_ready():
                print("Successfully connected to Weaviate!")
                return client
            else:
                print(f"Weaviate not ready, attempt {attempt + 1}/{max_retries}")
                
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise
    
    raise Exception("Failed to connect after all retries")

# Use the retry connection
# client = connect_with_retry()
# print(client.is_ready())