# # import os
# # file = "/root/TutorAI/backend/app/data/networks"
# # base_name = os.path.splitext(os.path.basename(file))[0]
# # print(base_name)
# from utils import parse
import weaviate

import os


# print(files)
# from weaviate.classes.query import MetadataQuery

# # file = "/root/TutorAI/backend/app/data/ComputerApplicationCommerce1Year"
client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                grpc_port=8081,
                skip_init_checks=True,
                additional_config=weaviate.classes.init.AdditionalConfig(
                    timeout=weaviate.classes.init.Timeout(init=30, query=60, insert=120)
                )
            )
collection_name = "Provisional_Certificate"

try:
    # Check if collection exists first (optional)
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
    else:
        print(f"Collection '{collection_name}' does not exist.")
except Exception as e:
    print(f"Error deleting collection: {e}")
finally:
    client.close()
            
collection = client.collections.get(collection_name)

# query = "Need for netwrok"
# res = collection.query.hybrid(query, limit=30, alpha=0.2, return_metadata=MetadataQuery(score=True, explain_score=True))

# res_objects = []
# for obj in res.objects:
#     if obj.metadata.score>0.3:
#         print("Section:",obj.properties['section'])
#         print("Chunk ID:",obj.properties['chunk_id'])
#         print("first: ",obj.metadata.score)
#         res_objects.append(obj.properties)

#     # print("first: ",obj.metadata.score)

# # client.close()
#     # print("second: ",obj.metadata.explain_score)

# # # Delete a collection


# import re
# from rapidfuzz import process

# file = "/root/TutorAI/backend/app/terms/netwroks_new.txt"

# with open(file,"r",encoding="utf-8") as f:
#     doc_terms = [line.strip() for line in f if line.strip()]

# def normalize_query(query,threshold=80):
#     words = re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
#     crcted_words = []

#     for w in words:
#         if w in doc_terms:
#             crcted_words.append(w)
#         else:
#             match = process.extractOne(w, doc_terms)
#             if match and match[1] >= threshold:
#                 crcted_words.append(match[0])
#             else:
#                 crcted_words.append(w)

#     return " ".join(crcted_words)

# from rapidfuzz import process
# import re
# from spellchecker import SpellChecker

# spell = SpellChecker()

# def normalize_query(query, threshold=80):
#     words = re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
#     crcted_words = []

#     for w in words:
#         if w in spell:  # ✅ valid English word
#             crcted_words.append(w)
#         else:  # maybe typo / abbreviation
#             match = process.extractOne(w, doc_terms)
#             if match and match[1] >= threshold:
#                 crcted_words.append(match[0])
#             else:
#                 # fallback: spellchecker suggestion
#                 suggestion = spell.correction(w)
#                 crcted_words.append(suggestion if suggestion else w)

#     return " ".join(crcted_words)


# import re
# from rapidfuzz import process

# def normalize_query(query, threshold=80):
#     words = re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
#     corrected_words = []

#     for w in words:
#         # ✅ if already valid English word, keep it
#         if w in spell:  
#             corrected_words.append(w)
#             continue

#         # 🔹 Step 1: Try spellchecker
#         suggestion = spell.correction(w)
#         if suggestion and suggestion in spell:
#             corrected_words.append(suggestion)
#             continue

#         # 🔹 Step 2: Fallback to doc terms fuzzy match
#         match = process.extractOne(w, doc_terms)
#         if match and match[1] >= threshold:
#             corrected_words.append(match[0])
#         else:
#             corrected_words.append(w)  # keep original if nothing works

#     return " ".join(corrected_words)


# query = "tell me abt com netwks"
# print(normalize_query(query))

# # with open(file,"r") as f:
# #     text = f.read()

# # # sections = parse(text)


# # max_len=float("-inf")
# # for obj in (chunk_objs):
# #     max_len = max(max_len, len(obj["text"]))
#     # if len(obj["text"])>=2700:
#     #     print("Chunk ID:", obj["chunk_id"])
#     #     print("Section:", obj["section"])
#     #     print("Text:", obj["text"], "\n")
#     #     break

# # print(max_len)
# # print(len("""o Octal to binary
# # o Hexadecimal to binary
# # o Octal to hexadecimal • Binary addition • Data representation
# # o Representation of numbers
# # o Representation of
# # o Representation of audio,
# # image and video
# # Computers have now become an integral part of our daily life. People use computers for a variety of reasons and purposes. Be it education, business, entertainment, communication, government service or transportation, computers are inevitable today. As far as students are concerned, computers are used for learning different subjects effectively and for carrying out learning activities apart from their primary functions of computing. Try to recollect the situations where we used computers and identify the benefits you got from it. Therefore it is essential to know more about computers and its applications. This chapter presents the concepts of data processing and functional units of computer. Different data representation methods used in computers are also discussed in this chapter. """))
# # for i, c in enumerate(chunk_objs):
# #     if "operating" in c["text"].lower() or "operating system" in c["text"].lower():
# #         print(i, c["chunk_id"], c["section"])
# #         print(c["text"][:400])
# #         print("\n")
# #         # break
# # else:
# #     print("No chunk contains the phrase — chunking produced no OS chunk.")


# import os
# from dotenv import load_dotenv
# import requests

# load_dotenv()

# invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"

# headers = {
#     "Authorization": f"Bearer {os.getenv('RERANKER_MODEL_API')}",
#     "Accept": "application/json",
# }

# payload = {
#   "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
#   "query": {
#     "text": "What is the GPU memory bandwidth of H100 SXM?"
#   },
#   "passages": [
#     {
#       "text": "The Hopper GPU is paired with the Grace CPU using NVIDIA's ultra-fast chip-to-chip interconnect, delivering 900GB/s of bandwidth, 7X faster than PCIe Gen5. This innovative design will deliver up to 30X higher aggregate system memory bandwidth to the GPU compared to today's fastest servers and up to 10X higher performance for applications running terabytes of data."
#     },
#     {
#       "text": "A100 provides up to 20X higher performance over the prior generation and can be partitioned into seven GPU instances to dynamically adjust to shifting demands. The A100 80GB debuts the world's fastest memory bandwidth at over 2 terabytes per second (TB/s) to run the largest models and datasets."
#     },
#     {
#       "text": "Accelerated servers with H100 deliver the compute power—along with 3 terabytes per second (TB/s) of memory bandwidth per GPU and scalability with NVLink and NVSwitch™."
#     }
#   ]
# }

# session = requests.Session()

# response = session.post(invoke_url, headers=headers, json=payload)

# response.raise_for_status()
# response_body = response.json()
# print(response_body)

# from utils import rerank

# query = "Computer Networks"

# docs = [
# {
#       "section": "8.1 Computer Network",
#       "text": "Computer network is a group of computers...",
#       "chunk_id": "8.1 Computer Network-1"
#     },
#     {
#       "section": "8.1.2 Some Key Terms",
#       "text": "Noise: Noise is unwanted electrical...",
#       "chunk_id": "8.1.2 Some Key Terms-3"
#     }
# ]

# print(rerank(query, docs))






