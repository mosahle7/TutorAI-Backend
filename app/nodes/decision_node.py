from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# import os
# from dotenv import load_dotenv

# load_dotenv()

sys_msg = """You are a query classifier.

RULES:
- If the query refers to past conversation or previous answers → respond exactly "memory"
- If the query refers to content in uploaded files, documents, or external content → respond exactly "document"
- If the query is vague/unclear/one word with no topic (like "wat", "yo", "how") → memory
- If "explain", "summarize" or "shorten" is followed by a topic word -> document 
- If "explain", "summarize" or "shorten" is alone or followed by pronouns ("it", "this", "that") → memory

EXAMPLES:
Query: "What did I just ask?" : memory
Query: "Explain abt networks" : document
Query: "Explain it" : memory
Query: "Write in points" : memory
Query: "WiFi" : document
Query: "Shorten" : memory
Query: "Shortnote on Nationalism" : document
Query: "Shorten Guided medium" : document
Query: "What" : memory
Query: "Wats Data": document
Query: "Tell abt WiFi" : document
Query: "Give it in a few paragraphs" : memory
Query: "Make it concise" : memory

Respond with ONLY one word: "memory" OR "document".
"""

def router(llm_model, query):
    messages = [SystemMessage(content="Reasoning Mode: OFF")]
    messages.append(HumanMessage(content=f"""{sys_msg}
Query: {query}"""))
    
    result = llm_model.invoke(messages)
    r = result.content.strip().lower()
    return r
