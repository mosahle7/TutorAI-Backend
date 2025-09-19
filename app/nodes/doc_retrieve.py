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
from ..config import client, collection, terms

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

def check_mode(llm_model, query):
    messages = [SystemMessage(content="Reasoning Mode: OFF")]

    sys_mgs = f"""You are tasked with classifying the mode of explanation of the given query.
You must classify the query into one of the following modes: "explanatory", "concise", "default".

RULES:
- If the query contains "explain", "describe", "long", "essay", "detail" → "explanatory"
- If the query contains words like "summarize", "short", "in brief", "concise", "main points", "bullet points" → "concise"
- Otherwise → "default"

EXAMPLES:
1. "What is something?" - default
2. "Explain something." - explanatory
3. "Summarize the main points of something." - concise
4. "Write about something in long" - explanatory
5. "Tell me about the process of something." - default
6. "What are benefits of something in short?" - concise
7. "Write about something" - default
8. "Something" - default
9. "Describe something" - explanatory

Respond with ONLY one word: explanatory, concise, or default.

Query: {query}
    """

    messages.append(HumanMessage(content=sys_mgs))

    response = llm_model.invoke(messages)
    mode = response.content.strip().lower()

    if "explanatory" in mode:
        return "explanatory"
    elif "concise" in mode:
        return "concise"
    else:
        return "default"
    
def mode_prompts(mode):
    if mode == "explanatory":
        prompt = f"""You are in Explanatory Mode. Provide detailed explanation with that covers: definition, context, examples and relevance.

RESPONSE RULES:
   - Do not explicitly label sections with these: definition, context, examples and relevance. Instead, organize the response into coherent paragraphs and, use your own suitable headings wherever needed.
   - If there is enough info in Retrieved Information, produce atleast 4-5 short paragraphs, each atleast 3–4 sentences.
   - If there is NOT enough info, answer as completely as possible.
   - Do NOT count Summary and Sources sections as Paragraphs.
   - Summary and Sources should be there.
   - Summary and Sources should come after Paragraphs.
   - Use Markdown format with ## Summary ## and ## Sources ## headings.

RESPONSE FORMAT:

[Paragraphs]

(After the paragraphs, add the Summary and Sources sections)

## Summary: ##
- [point 1]
- [point 2]
- .....

## Sources: ##
- [Section 1]
- [Section 2]
- .....
"""
    elif mode == "concise":
        prompt = f"""You are in Concise Mode. Provide brief, to-the-point answers with no unnecessary details.

RESPONSE RULES:
   - If there is enough info in Retrieved Information, answer in 1 paragraph.
   - If there is NOT enough info, answer as completely as possible.
   - Sources section does NOT count as Paragraphs.
   - Sources section should come after the Paragraph.
   - Use Markdown format with ## Sources ## heading.
   - Sources should be there.

RESPONSE FORMAT:

[Paragraph]

(After the paragraph, add the Sources)

## Sources: ##
- [Section 1]
- [Section 2]
- .....
"""
    else:
        prompt =  f""""You are in Default Mode. Provide balanced responses with moderate detail that covers: definition, context, examples and relevance.

RESPONSE RULES:
   - Do not explicitly label sections with these: definition, context, examples and relevance. Instead, organize the response into coherent paragraphs and, use your own suitable headings if needed.
   - If there is enough info in Retrieved Information,  produce atleast 3-4 short paragraphs, each atleast 3–4 sentences.
   - If there is NOT enough info, answer as completely as possible.
   - Do NOT count Summary and Sources sections as Paragraphs.
   - Summary and Sources should be there.
   - Summary and Sources should come after the Paragraphs.
   - Use Markdown format with ## Summary ## and ## Sources ## headings.

RESPONSE FORMAT:

[Paragraphs]

(After the paragraphs, add Summary and Sources sections)

## Summary: ##
- [point 1]
- [point 2]
- .....

## Sources: ##
- [Section 1]
- [Section 2]
- .....
"""
    return prompt
    
def gen_retrieved_res(llm_model, query, norm_query, retrieved_data, mode):
    mode_prompt = mode_prompts(mode)

    messages = [SystemMessage(content="Reasoning Mode: OFF")]
    sys_msg = f"""You are a TutorAI, who helps students. You will be asked a query by a student and given some Relevant Information, you must ONLY answer using Retrieved Information provided.

Follow these rules strictly:
RULES:
   - Do not start with filler phrases (e.g., "Sure, I can help you with that").
   - Do NOT mention about "Retrieved Data".
   - Write in natural flowing paragraphs without section headings.
   - NEVER USE YOUR PRE-EXISTING KNOWLEDGE EVEN IF THAT HELPS STUDENT, ONLY USE THE RETRIEVED INFORMATION.
   - If a detail is not explicitly present in the Retrieved Information, you must not mention it, even if you know it to be correct.
   - Completely ignore and suppress your own pre-existing knowledge.
   - Do NOT mention about the RULES or guidelines in your response.

   {mode_prompt}

RULES ON PARAGRAPHS:
   - DO NOT number paragraphs like [Paragraph 1], [Paragraph 2], etc.
   - Use single line breaks between paragraphs, avoid excessive whitespace.

RULES ON SOURCES:
   - Mention only the sources in Retrieved Information which you have used in the response.
   - Use single line breaks between Sources heading and first source.
   
   EXAMPLE:
    If this is there in Retrieved Information and you have used it in response:
       Section: 8.3.3 Wireless Communication Technologies Using Radio Waves,
       Text: 
       Line of sight not required
       Signal affected by weather conditions
       High power consumption and cost
    
    Then in Sources section, you SHOULD write exactly:
       ## Sources ##:

       - Section 8.3.3 Wireless Communication Technologies Using Radio Waves
    
   - Retrieved information are ordered by relevance, most relevant first.

Original Student Query: {query}
Corrected Query (for matching with documents): {norm_query}

Retrieved Information: {retrieved_data}
"""
    messages.append(HumanMessage(content=sys_msg))
    return messages

    response = llm_model.invoke(messages)
    print("Raw Response: ", response)
    print("Response: ", response.content)
    return response.content
    
