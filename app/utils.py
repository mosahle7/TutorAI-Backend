import re
from typing import List, Dict

def parse(text):
    sections = {}
    pattern = re.compile(r'^(\d+(\.\d+)*)\s+(.*)')
    sec = "Introduction"
    sections[sec]=""
    lines = text.split("\n")
    for line in lines:
        match = pattern.match(line)
        if match:
            sec = line
        else:
            if sec in sections:
                sections[sec]+="\n"+line
            else:
                sections[sec]=""
                sections[sec]+=line
    return sections

def chunking(sections):
    chunk_objs=[]
    min_len=200
    max_len=500

    for sec in sections:
        paras = sections[sec].split("\n")
        new_para=""
        num_chunk=0
        sec_num = sec.split(" ")[0]
        num_paras=len(paras)
        num=0
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
                num_chunk+=1
                chunk_obj = {
                    "chunk_id":f"{sec_num}-{num_chunk}",
                    "section":sec,
                    "text":new_para
                }
                new_para=""
                chunk_objs.append(chunk_obj)
                continue
            if not para:
                continue
            if new_para=="":
                new_para+=para
            else:
                new_para+='\n'+para
            if num == num_paras:
                num_chunk+=1
                if len(chunk_objs[-1]["text"]+para)<max_len:
                    chunk_objs[-1]["text"]+="\n"+para
                else:
                    chunk_obj = {
                        "chunk_id":f"{sec_num}-{num_chunk}",
                        "section":sec,
                        "text":new_para
                    }
                    chunk_objs.append(chunk_obj)
    return chunk_objs


def gen_single_ip(
    llm_model,
    messages: List[Dict[str, str]],
    # role: str = "user",
    top_p: float = 0.7,
    temperature: float = 0.5,
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
            **kwargs,
        }

        try:
            completion = llm_model.chat.completions.create(**payload)
            msg = completion.choices[0].message  # ChatCompletionMessage
            return msg.content
        
        except Exception as e:
            print(f"Errror generating response: {e}")
            return {"error":str(e)}

def hybrid_search(collection,query,top_k=7,alpha=0.5):
    try:
        res = collection.query.hybrid(query,limit=top_k,alpha=alpha)
        res_objs = [x.properties for x in res.objects]
        
        # for obj in res_objs:
        #     s=""
        #     for key in obj.keys():
        #         s+=f"{key}: {obj[key]}\n"
        #     print(s)
        return res_objs
    except Exception as e:
        print(f"Failed retrieving information: {e}")
        return {"error":str(e)}
    

def gen_final_response(llm_model,collection,query:str):
    top_k_docs = hybrid_search(collection, query)
    formatted_data = ""

    for doc in top_k_docs:
        doc_layout = (
            f"Section: {doc['section']},"
            f"Text: {doc['text']}"
        )

        formatted_data += doc_layout+"\n"+"\n"
    
    retrieved_data = formatted_data

    messages = [
            {
                "role": "system", "content": f"""You are a Tutor. You will be asked a query by a student and given some relevant textbook information.

                - Answer the query **only using the provided information** in a clear, natural language.
                    If user does not mention about the length of response (default) or if users says "enriched", "long" or "detail", follow this:
                        - Always produce atleast 3–4 short paragraphs (with paragraphs having atleast 3–4 sentences each).
                        - Cover multiple aspects (definition, context, examples, relevance).
                        - Never collapse the answer into 1 paragraph unless the user explicitly requests a concise/short/brief/summary.
                    Concise mode:
                        - If the user explicitly says "concise", "short", "brief", or "summary", then respond in only 1–2 sentences.

                - Do not start with filler phrases (e.g., "Sure, I can help you with that").
                - If the answer cannot be found in the retrieved information, reply exactly: "The textbook does not provide this information."  
                - Do not use outside knowledge, even if you know it and it helps the student.
                - Do not add any commentary or meta explanation.
                - At the end of your answer, list the sources you used, citing only their section names in parentheses.

                Retrieved Information: {retrieved_data}
                """
            },
            {
                "role":"user", "content":f"Student Query: {query}"
            }
     ]

    res = gen_single_ip(llm_model,messages,max_tokens=2000)
    print(res)
    return res