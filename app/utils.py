import json
import re
from typing import List, Dict
import fitz
import numpy as np
from sklearn.cluster import KMeans
import os, re
import requests
from collections import Counter
from weaviate.classes.query import MetadataQuery
from rapidfuzz import process
import re
from spellchecker import SpellChecker

spell = SpellChecker()

def list_files():
    docs_dir = "/root/TutorAI/backend/app/data"
    files = [f for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir,f))]
    return files

def is_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
            return header == b"%PDF-"
    except:
        return False

def extract_pdf_text(pdf_path, output_txt_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        blocks = [b for b in blocks if b[6] == 0]  # keep only text blocks

        # Get x positions (left edge of block)
        x_positions = np.array([[b[0]] for b in blocks])

        # Default: assume single column
        column_labels = np.zeros(len(blocks), dtype=int)

        # Try detecting 2 columns via KMeans
        if len(blocks) > 6:  # only try if enough blocks
            try:
                kmeans = KMeans(n_clusters=2, n_init="auto").fit(x_positions)
                column_labels = kmeans.labels_

                # Check distance between clusters
                centers = sorted(kmeans.cluster_centers_.flatten())
                if abs(centers[1] - centers[0]) < 50:  
                    # very close, treat as one column
                    column_labels = np.zeros(len(blocks), dtype=int)
            except:
                pass  # fallback to single column

        # Group blocks by column
        col_blocks = {c: [] for c in set(column_labels)}
        for label, block in zip(column_labels, blocks):
            col_blocks[label].append(block)

        # Sort: left column first (min x), then right
        sorted_cols = sorted(col_blocks.items(), key=lambda kv: min(b[0] for b in kv[1]))

        for _, col in sorted_cols:
            # Sort top-to-bottom inside each column
            for b in sorted(col, key=lambda b: (b[1], b[0])):
                text = b[4].strip()
                # 🔑 Clean up line breaks inside block (paragraph stays intact)
                text = re.sub(r"\n+", " ", text).strip()
                if text:
                    all_text.append(text)

    # Join blocks with double newline to separate paragraphs
    clean_text = "\n\n".join(all_text)

    # Save output
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    return output_txt_path

def parse(text):
    sections = {}
    # pattern = re.compile(r'^(\d+(\.\d+)*)\s+(.*)')
    sec = "Introduction"
    sections[sec]=""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)  # shrink 3+ blank lines to 2

    numbered = r"\d+(?:\.\d+)*\s+.+"   # matches "1", "1.1 Title", "2.3.4 Something"
    named = r"(?:[A-Z][A-Z0-9 .-]{3,}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"  # matches "ELECTRICAL MACHINES", "Data Structures"
    section_pattern = rf"^({numbered}|{named})$"

    text = re.sub(r"(?<!\n)\n?(" + section_pattern + r")(?!\n)", r"\n\1\n", text, flags=re.MULTILINE)
    header_re = re.compile(section_pattern, flags=re.MULTILINE)

    buf = []

    def flush_buf():
        nonlocal buf, sec
        chunk = "\n".join(buf).strip()
        chunk = re.sub(r"\n{3,}","\n\n", chunk)

        sections[sec] = (
            sections.get(sec,"").strip()
            + ("\n\n" if sections.get(sec,"").strip() and chunk else "")
            + chunk
        ).strip()

        buf = []
    
    for raw in text.split("\n"):
        line = raw.strip()

        # skip lines that are just numbers or outline numbers like "2.4.1"
        if re.fullmatch(r"\d+(\.\d+)*", line):
            continue

        if header_re.fullmatch(line):
            flush_buf()
            sec = line
            sections.setdefault(sec, "")
            continue

        if line == "":
            buf.append("")
            continue

        if not re.match(r"^\s*(?:[-\u2022\u25CF\u00B7•◦∙]|\d+\.)\s", line):
            line = re.sub(r"[ \t]{2,}", " ", line)
        buf.append(line)


    flush_buf()

    if sections.get("Introduction", "").strip() == "" and len(sections) > 1:
        sections.pop("Introduction", None)
    
    return sections

    # lines = text.split("\n")
    # for line in lines:
    #     match = pattern.match(line)
    #     if match:
    #         sec = line
    #     else:
    #         if sec in sections:
    #             sections[sec]+="\n"+line
    #         else:
    #             sections[sec]=""
    #             sections[sec]+=line
    # return sections

def split_by_sent(text, max_len=800):
    sentences = re.split(r'(?<=[.!?])\s+', text) 
    chunks, buf, size = [], [], 0

    for sent in sentences:
        if len(sent)>max_len:
            if buf:
                chunks.append(" ". join(buf))
                buf, size = [], 0
            chunks.append(sent)
            continue
        
        if size + len(sent) + 1>max_len:
            chunks.append("". join(buf))
            buf, size = [], 0
        
        buf.append(sent)
        size += len(sent) + 1

    if buf:
        chunks.append(" ". join(buf))

    return chunks

def chunking(sections):
    chunk_objs=[]
    min_len=400
    max_len=800

    
    for sec in sections:
        paras = sections[sec].split("\n")
        new_para=""
        # sec_num = sec.split(" ")[0]
        num_paras=len(paras)
        num=0
        num_chunk=0
        for para in paras:
            num+=1
            # if len(new_para+para)>=max_len:
            #     num_chunk+=1
            #     chunk_obj = {
            #         "chunk_id":f"{sec_num}-{num_chunk}",
            #         "section":sec,
            #         "text":new_para+"\n"+para
            #     }
            #     new_para=""
            #     chunk_objs.append(chunk_obj)
            #     continue
            if len(new_para)<min_len and num!=num_paras:
                if not para:
                    continue
                if len(new_para)==0:
                    new_para+=para
                else:
                    new_para+="\n"+para
                continue

            elif len(new_para)>=min_len:
                for piece in split_by_sent(new_para):
                    num_chunk+=1
                    chunk_obj = {
                        "chunk_id":f"{sec}-{num_chunk}",
                        "section":sec,
                        "text":piece
                    }
                    chunk_objs.append(chunk_obj)
                new_para=""
                continue
                # num_chunk+=1
                # chunk_obj = {
                #     "chunk_id":f"{num_chunk}",
                #     "section":sec,
                #     "text":new_para
                # }
                # new_para=""
                # chunk_objs.append(chunk_obj)
                # continue
            if not para:
                continue
            if new_para=="":
                new_para+=para
            else:
                new_para+='\n'+para
            if num == num_paras:
                # num_chunk+=1
                # if chunk_objs and len(chunk_objs[-1]["text"]+para)<max_len:
                #     chunk_objs[-1]["text"]+="\n"+para
                # else:
                #     chunk_obj = {
                #         "chunk_id":f"{num_chunk}",
                #         "section":sec,
                #         "text":new_para
                #     }
                #     chunk_objs.append(chunk_obj)
                for piece in split_by_sent(new_para):
                    num_chunk+=1
                    chunk_obj = {
                        "chunk_id":f"{sec}-{num_chunk}",
                        "section":sec,
                        "text":piece
                    }
                    chunk_objs.append(chunk_obj)
    return chunk_objs

def build_doc_terms(chunk_objs, top_n=500):
    text_corpus = " ".join(obj["text"] for obj in chunk_objs).lower()
    tokens = re.findall(r"\b[a-z]{3,}\b", text_corpus)
    common_terms = [w for w,_ in Counter(tokens).most_common(top_n)]
    return list(set(common_terms))


def normalize_query(query,terms,threshold=80):
    words = re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
    crcted_words = []

    for w in words:
        if w in spell:  # ✅ valid English word
            crcted_words.append(w)
        else:  # maybe typo / abbreviation
            match = process.extractOne(w, terms)
            if match and match[1] >= threshold:
                crcted_words.append(match[0])
            else:
                # fallback: spellchecker suggestion
                suggestion = spell.correction(w)
                crcted_words.append(suggestion if suggestion else w)

    return " ".join(crcted_words)

def gen_single_ip(
    llm_model,
    messages: List[Dict[str, str]],
    # role: str = "user",
    top_p: float = 0.75,
    temperature: float = 0,
    max_tokens: int = 5000,
    model: str = "nvidia/llama-3.1-nemotron-nano-8b-v1",
    **kwargs,
):
        payload = {
            "model": model,
            # "messages": [{"role":"user","content":query}],
            "messages":messages,
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream":True,
            **kwargs,
        }

        try:
            completion = llm_model.chat.completions.create(**payload)

            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    # print(f"Yielding: {repr(content)}")  # Add this debug line
                    yield content

        except Exception as e:
            yield f"Error: {str(e)}"
        # msg = completion.choices[0].message  # ChatCompletionMessage
        # return msg.content
        
        # except Exception as e:
        #     print(f"Errror generating response: {e}")
        #     return {"error":str(e)}

def hybrid_search(collection,query,terms,top_k=30,alpha=0.2):
    try:
        norm_query = normalize_query(query,terms)
        print(norm_query)
        res = collection.query.hybrid(norm_query,limit=top_k,alpha=alpha,return_metadata=MetadataQuery(score=True, explain_score=True))
        res_objs = []
        for obj in res.objects:
            if obj.metadata.score>0.25:
                res_objs.append(obj.properties)    

        res = [obj for obj in res.objects if obj.metadata.score>0.3]   
        # for obj in res_objs:
        #     s=""
        #     for key in obj.keys():
        #         s+=f"{key}: {obj[key]}\n"
        #     print(s)
        return res, res_objs
    except Exception as e:
        print(f"Failed retrieving information: {e}")
        return {"error":str(e)},[]
    
def check_mode(llm_model, query):
    messages = [
        {
            "role":"system", "content": f"""You will be given a query, your task is to identify the mode of explanation of the query.
You ONLY need to respond with one of these words: "default", "explanatory" or "concise".

Examples:
1. "What is something?" - default
2. "Explain something." - explanatory
3. "Summarize the main points of something." - concise
4. "Write about something in long" - explanatory
5. "Tell me about the process of something." - default
6. "What are benefits of something in short?" - concise
"""
        },
        {
            "role":"user", "content":query
        }
    ]
    
    payload = {
        "model": "nvidia/llama-3.1-nemotron-nano-8b-v1",
        "messages": messages,
        "top_p": 0.7,
        "temperature": 0,
        "max_tokens": 3,
    
    }
    completion = llm_model.chat.completions.create(**payload)

    msg = completion.choices[0].message.content.strip().lower()  # ChatCompletionMessage

    if "explanatory" in msg:
        return "explanatory"
    elif "concise" in msg:
        return "concise"
    else:
        return "default"
    
def prompts_mode(mode):
    if mode == "default":
        prompt = f"""- Try to produce at least 3–4 short paragraphs (with paragraphs having at least 3–4 sentences each), only if enough information is available in Retrieved Information.
- Cover multiple aspects (definition, context, examples, relevance) within paragraphs.
- If there is not enough information in Retrieved Information, answer as completely as possible.
- Do not include a summary for responses with fewer than 3 paragraphs.
"""
    elif mode == "explanatory":
        prompt = f"""- Always produce atleast 4-5 short paragraphs (with paragraphs having atleast 3–4 sentences each) only if enough information is available in Retrieved Information.
- Cover multiple aspects (definition, context, examples, relevance).
"""
    else:
        prompt=f"""- Always answer in 1 paragraph, containing 3–4 sentences only if enough information is available in Retrieved Information.
"""
    return prompt


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



def gen_final_response(llm_model,collection,query:str,terms):
    initail_res,initial_docs = hybrid_search(collection, query, terms)
    if initial_docs == []:
        yield f"The uploaded document does not provide any information about {query}.".replace("\n","")
        return
    
    mode = check_mode(llm_model, query)
    mode_prompt = prompts_mode(mode)
    top_k_docs = rerank(query, initail_res, initial_docs)

    formatted_data = ""

    for idx,doc in enumerate(top_k_docs, start=1):
        doc_layout = (
            f"Rank {idx} | Section: {doc.properties['section']},"
            f"Text: {doc.properties['text']}"
        )

        formatted_data += doc_layout+"\n\n"
    
    retrieved_data = formatted_data

    print(retrieved_data)

    messages = [
            {
                "role": "system", "content": f"""You are a TutorAI, who helps students. You will be asked a query by a student and given some relevant textbook information, you must ONLY answer using Retrieved Information provided.

{mode_prompt}

- Do not start with filler phrases (e.g., "Sure, I can help you with that").
- Write in natural flowing paragraphs without section headings.
- If something is not explicitly stated in the Retrieved Information, you MUST NOT mention it, even if you know it is true.
- Use single line breaks between paragraphs, avoid excessive whitespace.
- If your response contains 3 or more paragraphs, end with a summary section.
- End your response in this exact format:

[Paragraphs]

**Summary:** (only if 3+ paragraphs)
- [point 1]
- [point 2]
- [point 3]

**Sources:**
- [Section 1]
- [Section 2]

- Do not add blank lines between heading and section names in Sources section and ONLY use the sources which you have used in your response.
- Do not inlude TEXT or RANK in Sources section, ONLY Section needed.
- Do not include any meta-commentary about the response format, guidelines or your thinking steps in the response.
- DO NOT number paragraphs like [Paragraph 1], [Paragraph 2], etc.
- Completely ignore and suppress your own pre-existing knowledge.
- NEVER USE YOUR PRE-EXISTING KNOWLEDGE EVEN IF THAT HELPS STUDENT, ONLY USE THE RETRIEVED INFORMATION.
- Retrieved information are ordered by relevance, most relevant first.

Retrieved Information: {retrieved_data}
"""
            },
            {
                "role":"user", "content":f"Student Query: {query}"
            }
     ]

    # res = gen_single_ip(llm_model,messages,max_tokens=2000)
    # print(res)
    # return res

    yield from gen_single_ip(llm_model,messages,max_tokens=2000)


