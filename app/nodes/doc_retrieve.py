import os
import weaviate
import time
import tqdm
import requests
import re
from spellchecker import SpellChecker
from rapidfuzz import process
from weaviate.classes.query import MetadataQuery
from langchain_core.messages import SystemMessage, HumanMessage
from symspellpy import SymSpell

def initialize_client(max_retries=5, delay=2):
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

collection_name = "netwroks_new"

def initialize_collection(client):
    collection = client.collections.get(collection_name)

    with open("/root/TutorAI/backend/app/terms/netwroks_new.txt", "r", encoding="utf-8") as f:
        terms = [line.strip() for line in f if line.strip()]
    return collection, terms

client = initialize_client()
collection, terms = initialize_collection(client)

spell = SpellChecker()

def build_symspell(prefix_length=7):
    # ✅ only prefix_length is accepted in init
    sym_spell = SymSpell(prefix_length=prefix_length)

    for term in terms:
        sym_spell.create_dictionary_entry(term, 1)

    return sym_spell

def normalize_query(query, sym_spell=None, threshold=80):
    words = re.findall(r"[a-zA-Z0-9\-_/\.]{2,}", query.lower())
    corrected = []

    for w in words:
        if w in terms:  # ✅ exact doc match
            corrected.append(w)
        elif w in spell:  # ✅ valid English word
            corrected.append(w)
        else:
            # Try SymSpell first (typo correction)
            if sym_spell:
                suggestions = sym_spell.lookup(w, verbosity=2)
                if suggestions:
                    corrected.append(suggestions[0].term)
                    continue

            # Fallback: fuzzy match against domain terms
            match = process.extractOne(w, terms)
            if match and match[1] >= threshold:
                corrected.append(match[0])
            else:
                corrected.append(w)  # last resort: leave as-is

    return " ".join(corrected)



def hybrid_search(norm_query):
    try:
        print(norm_query)
        res = collection.query.hybrid(norm_query,limit=30,alpha=0.2,return_metadata=MetadataQuery(score=True, explain_score=True))
        res_objs = []
        for obj in res.objects:
            if obj.metadata.score>0.25:
                res_objs.append(obj.properties)    

        res = [obj for obj in res.objects if obj.metadata.score>0.25]   
        # for obj in res_objs:
        #     s=""
        #     for key in obj.keys():
        #         s+=f"{key}: {obj[key]}\n"
        #     print(s)
        return res, res_objs
    

    except Exception as e:
        print(f"Failed retrieving information: {e}")
        return {"error":str(e)},[]
    
def rerank(query, initial_res, docs):
    documents = [doc['text'] for doc in docs]
    top_k=20
    try:
        rerank_res = requests.post(
            'http://127.0.0.1:8000/weaviate/rerank',
            json={'query': query, 'documents': documents},
            timeout=30
        )

        if rerank_res.status_code != 200:
            raise RuntimeError(f"Rerank API returned {rerank_res.status_code}")

        rerank_data = rerank_res.json()
        reranked_scores = rerank_data['scores']

        score_map = {s['document']: s['score'] for s in reranked_scores}

        # 4. Attach scores and sort
        combined_results = [
            (obj, score_map.get(obj.properties['text'], 0.0))
            for obj in initial_res
        ]
        combined_results.sort(key=lambda x: x[1], reverse=True)

        # 5. Take top_k and return properties list
        response_objects = [obj for obj,_ in combined_results[:top_k]]
        print("Reraking Done!")
        return response_objects

    except Exception as e:
        print(f"Reranking failed: {e}")
        # Fallback to original ordering
        return [obj for obj in initial_res[:top_k]]


def gen_retrieved_res(llm_model, query, norm_query, retrieved_data):
    messages = [SystemMessage(content="Reasoning Mode: OFF")]

    sys_msg = f"""You are a TutorAI, who helps students. You will be asked a query by a student and given some relevant textbook information, you must ONLY answer using Retrieved Information provided.

RULES:
- Do not start with filler phrases (e.g., "Sure, I can help you with that").
- Do NOT mention about "Retrieved Data".
- Write in natural flowing paragraphs without section headings.
- NEVER USE YOUR PRE-EXISTING KNOWLEDGE EVEN IF THAT HELPS STUDENT, ONLY USE THE RETRIEVED INFORMATION.
- If a detail is not explicitly present in the Retrieved Information, you must not mention it, even if you know it to be correct.
- Completely ignore and suppress your own pre-existing knowledge.
- Do NOT mention about the instructions or guidelines in your response.

RESPONSE FORMAT:

[Paragraphs]

**Summary:** (only if 3+ paragraphs)
- [point 1]
- [point 2]
- .....

**Sources:**
- [Section 1]
- [Section 2]
- .....

ADDITIONAL RULES:
- DO NOT number paragraphs like [Paragraph 1], [Paragraph 2], etc.
- Use single line breaks between paragraphs, avoid excessive whitespace.
- If your response contains 3 or more paragraphs, end with a summary section.
- Do not add blank lines between heading and section names in Sources section and ONLY use the sources which you have used in your response.
- In Sources section, ONLY list "Section" that was used in response mentioned as in Retrieved Information, NO need of "Text". Do not invent sources. Only mention Sources once.
- Retrieved information are ordered by relevance, most relevant first.

Original Student Query: {query}
Corrected Query (for matching with documents): {norm_query}

Retrieved Information: {retrieved_data}
"""
    
    messages.append(HumanMessage(content=sys_msg))
    
    response = llm_model.invoke(messages)
    return response.content
    
