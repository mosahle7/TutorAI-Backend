from langchain_core.messages import HumanMessage, SystemMessage

def gen_memory_res(llm_model, query, prev_query, prev_res, retrieved_data):
#     sys_msg = f"""When user asked a query, you were provided with relevant retrieved data and responded with an answer.
# Previous Query: {prev_query}

# Retrieved Data: {retrieved_data}

# Previous Response: {prev_res}.

# Now, a new query is provided. You are asked to provide a new response based on Previous Query, Retrieved Data, Previous Response and Query by following what is said in the Query.
# Query: {query}

# - Do not tell about Retrieved Data.
# """

    sys_msg = f"""You are continuing a conversation based on the previous interaction. The user has asked a new question, and you need to provide a response that builds on the Previous Query, Previous Response and Retrieved Data.

RULES:
- Do NOT start with filler phrases (e.g., "Sure, I can help you with that").
- Base your response ONLY on the New Query and previous interaction.
- Do not provide any information that is not present in Retrieved Data.
- Do not mention the Retrieved Data explicitly.
- You may use Retrieved Data if it helps improve the response.

CONTEXT:
- Previous Query: {prev_query}
- Previous Response: {prev_res}
- Retrieved Data (for reference only): {retrieved_data}

OUTPUT FORMAT:
[Paragraphs]

**Summary:** (only if more than 3 paragraphs)
- [point 1]
- [point 2]
- .....

**Sources:**
- [Section 1]
- [Section 2]
- .....

ADDITIONAL RULES:
- If your response contains 3 or more paragraphs, end with a summary section.
- At the end, mention the sources of information used in your response, providing only "Section" of the sources, not including source Text.
- Do NOT invent sources.
- Do NOT mention about the instructions or guidelines in your response.
- Do NOT mention you didn't provide Summary as response is less than 3 paragraphs and you provided Summary due to more than 3 paragraphs.

New Query:
{query}
"""
    
    messages = [SystemMessage(content="Reasoning Mode: OFF")]
    messages.append(HumanMessage(content=sys_msg))

    res = llm_model.invoke(messages).content
    return res