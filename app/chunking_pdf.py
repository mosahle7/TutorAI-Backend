import re

file = "/root/TutorAI/backend/app/data/networks"

with open(file,"r") as f:
    text = f.read()

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

        if header_re.fullmatch(line.strip()):
            flush_buf()
            sec = line.strip()
            sections.setdefault(sec, "")
            continue

        if line.strip() == "":
            buf.append("")
            continue

        if not re.match(r"^\s*(?:[-\u2022\u25CF\u00B7•◦∙]|\d+\.)\s", line):
            line = re.sub(r"[ \t]{2,}", " ", line)
        buf.append(line.strip())

    flush_buf()

    if sections.get("Introduction", "").strip() == "" and len(sections) > 1:
        sections.pop("Introduction", None)
    
    return sections

def chunking(sections):
    chunk_objs=[]
    min_len=200
    max_len=500
    num_chunk=0
    
    for sec in sections:
        paras = sections[sec].split("\n")
        new_para=""
        # sec_num = sec.split(" ")[0]
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
                    "chunk_id":f"{num_chunk}",
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
                if chunk_objs and len(chunk_objs[-1]["text"]+para)<max_len:
                    chunk_objs[-1]["text"]+="\n"+para
                else:
                    chunk_obj = {
                        "chunk_id":f"{num_chunk}",
                        "section":sec,
                        "text":new_para
                    }
                    chunk_objs.append(chunk_obj)
    return chunk_objs
sections = parse(text)
chunk_objs = chunking(sections)

m=0
# for sec in sections:
#     print(sec,":\n")
#     print(sections[sec],"\n")
#     m+=1
#     if m==3:
#         break

print(len(chunk_objs,"\n"))
for obj in chunk_objs:
    print("Chunk ID: ",obj["chunk_id"])
    print("Section: ",obj["section"])
    print(obj["text"],"\n")
    m+=1
    if m==5:
        break
