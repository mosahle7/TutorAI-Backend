import os, getpass
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from .nodes.decision_node import router
from .nodes.doc_retrieve import build_symspell, normalize_query, hybrid_search, rerank, check_mode, gen_retrieved_res
from .nodes.memory_retrieve import gen_memory_res
from .nodes.question_llm_call import gen_questions

load_dotenv() 

# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")

# _set_env("LANGSMITH_API_KEY")
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_PROJECT"] = "tutorai"

llm_model = ChatNVIDIA(
    base_url="https://integrate.api.nvidia.com/v1",
    # api_key=os.getenv("MODEL_API"),
    # model="nvidia/llama-3.1-nemotron-nano-8b-v1",
    api_key=os.getenv("LLAMA_API"),
    model="meta/llama-3.1-70b-instruct",
    temperature=0,
    top_p=0.75,
    max_retries=3,
    max_completion_tokens=5000
)

class MsgState(MessagesState):
    retrieved_data : str
    mode : str

class MsgState2(MessagesState):
    retrieved_data : str
    num_questions : int

config = {"configurable": {"thread_id":"1"}}

def route_condition(state: MsgState) -> str:
    """Condition function to decide next node."""
    print("🤖 route_condition node")
    query = state["messages"][-1].content.lower()
    r = router(llm_model, query) 
    if r == "document":
        print("Router → document_retrieval")
        return "document_retrieval"
    print("Router → memory_retrieval")
    return "memory_retrieval"

def document_retrieval(state: MsgState):
    print("🤖 document_retrieval node")
    query = state["messages"][-1].content
    sym_spell = build_symspell()
    norm_query = normalize_query(query, sym_spell)

    initial_res, initial_docs = hybrid_search(norm_query)
    
    top_k_docs = rerank(norm_query, initial_res, initial_docs)

    formatted_data = ""

    for idx,doc in enumerate(top_k_docs, start=1):
        doc_layout = (
            f"Section: {doc.properties['section']},\n"
            f"Text:\n{doc.properties['text']}"
        )

        formatted_data += doc_layout+"\n\n"
    
    retrieved_data = formatted_data

    state["retrieved_data"] = retrieved_data
    return {"retrieved_data": state["retrieved_data"]}

def llm_retrieved_call(state: MsgState):
    print("🤖 llm_retrieved_call node")

    query = state["messages"][-1].content
    sym_spell = build_symspell()
    norm_query = normalize_query(query, sym_spell)
    mode = state["mode"]
    print("Mode: ",mode)

    retrieved_data = state["retrieved_data"]
    if retrieved_data is None or retrieved_data == "":
        state["messages"].append(AIMessage(content=f"I couldn't find any information about: {query}. Please try asking something else."))
        return {"messages": state["messages"][-1:]}
    
    # print(retrieved_data)
    msg = gen_retrieved_res(llm_model, query, norm_query, retrieved_data, mode)

    res_chunks = []
    chunk_count = 0

    for chunk in llm_model.stream(msg):
        if chunk.content:
            chunk_count += 1
            res_chunks.append(AIMessage(content=chunk.content))

    print(f"🤖 Collected {len(res_chunks)} chunks, returning to graph")

    # state["messages"].append(AIMessage(content="".join([c.content for c in res_chunks])))
    return {"messages": " ".join([c.content for c in res_chunks])}

def memory_retrieval(state: MsgState):
    if len(state["messages"])<3:
        print("Just starting conversation")
        state["messages"].append(AIMessage(content="Hi! This seems to be our first chat. What topic would you like to discuss?"))
        return {"messages": state["messages"][-1:]}
    
    else:
        query = state["messages"][-1].content
        prev_query = state["messages"][-3].content
        prev_res = state["messages"][-2].content
        retrieved_data = state["retrieved_data"]

        msg = gen_memory_res(llm_model, query, prev_query, prev_res, retrieved_data)

        res_chunks = []
        chunk_count = 0

        for chunk in llm_model.stream(msg):
            if chunk.content:
                chunk_count += 1
                res_chunks.append(AIMessage(content=chunk.content))

        print(f"🤖 Collected {len(res_chunks)} chunks, returning to graph")

        state["messages"].append(AIMessage(content="".join([c.content for c in res_chunks])))
        return {"messages": " ".join([c.content for c in res_chunks])}


    # state["messages"].append(AIMessage(content=res))
    # return {"messages": state["messages"][-1:]}

memory = MemorySaver()

builder = StateGraph(MsgState)

# builder.add_node("llm",llm_call)
builder.add_node(document_retrieval)
builder.add_node(memory_retrieval)
builder.add_node(llm_retrieved_call)
# builder.add_node("router", router)

# builder.add_edge(START, "router")
builder.add_conditional_edges(START, route_condition, ["document_retrieval", "memory_retrieval"])
builder.add_edge("document_retrieval", "llm_retrieved_call")
builder.add_edge("llm_retrieved_call", END)
builder.add_edge("memory_retrieval", END)

graph = builder.compile(checkpointer=memory)
# graph = builder.compile()

def question_generation(state: MsgState2):
    print("🤖 question_generation node")

    topic = state["messages"][-1].content
    sym_spell = build_symspell()
    norm_topic = normalize_query(topic, sym_spell)

    retrieved_data = state["retrieved_data"]
    if retrieved_data is None or retrieved_data == "":
        state["messages"].append(AIMessage(content=f"I couldn't generate any questions as the uploaded document does not provide any information on: {topic}."))
        return {"messages": state["messages"][-1:]}

    num_questions = state["num_questions"]

    questions = gen_questions(llm_model, topic, norm_topic, retrieved_data, num_questions)

    state["messages"].append(AIMessage(content=questions))
    return {"messages": state["messages"][-1:]}

builder2 = StateGraph(MsgState2)

builder2.add_node(document_retrieval)
builder2.add_node(question_generation)

builder2.add_edge(START, "document_retrieval")
builder2.add_edge("document_retrieval", "question_generation")
builder2.add_edge("question_generation", END)

graph2 = builder2.compile()

