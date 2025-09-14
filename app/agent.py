import os, getpass
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from .nodes.decision_node import router
from .nodes.doc_retrieve import initialize_client, initialize_collection, build_symspell, normalize_query,hybrid_search, rerank, gen_retrieved_res
from .nodes.memory_retrieve import gen_memory_res
load_dotenv()

collection_name = "networks"

client = initialize_client()
collection = initialize_collection(client)

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "tutorai"

llm_model = ChatNVIDIA(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("MODEL_API"),
    model="nvidia/llama-3.1-nemotron-nano-8b-v1",
    temperature=0,
    top_p=0.75,
    max_retries=3,
    max_completion_tokens=5000
)

# def llm_call(state: MessagesState):
#     msg = state["messages"][-1].content

#     if len(msg)<5:
#         print("Just END\n\n")
#         return {"route_to": END}
#     return {"messages": [llm_model.invoke(state["messages"])]}


class MsgState(MessagesState):
    retrieved_data : str

config = {"configurable": {"thread_id":"1"}}

# def router(state: MessagesState):
#     """Router node: doesn't decide where to go, just returns state."""
#     return {}   # ✅ must return a dict


def route_condition(state: MsgState) -> str:
    """Condition function to decide next node."""
    query = state["messages"][-1].content.lower()
    r = router(llm_model, query) 
    if r == "document":
        print("Router → document_retrieval")
        return "document_retrieval"
    print("Router → memory_retrieval")
    return "memory_retrieval"

    
def document_retrieval(state: MsgState):
    query = state["messages"][-1].content
    sym_spell = build_symspell()
    norm_query = normalize_query(query, sym_spell)
    initial_res, initial_docs = hybrid_search(norm_query)

    top_k_docs = rerank(norm_query, initial_res, initial_docs)

    formatted_data = ""

    for idx,doc in enumerate(top_k_docs, start=1):
        doc_layout = (
            f"Section: {doc.properties['section']},"
            f"Text: {doc.properties['text']}"
        )

        formatted_data += doc_layout+"\n\n"
    
    retrieved_data = formatted_data

    state["retrieved_data"] = retrieved_data
    return {"retrieved_data": state["retrieved_data"]}

def llm_retrieved_call(state: MsgState):
    query = state["messages"][-1].content
    sym_spell = build_symspell()
    norm_query = normalize_query(query, sym_spell)

    retrieved_data = state["retrieved_data"]
    if retrieved_data is None or retrieved_data == "":
        state["messages"].append(AIMessage(content=f"The uploaded document does not provide any information about: {query}"))
        return {"messages": state["messages"][-1:]}
    res = gen_retrieved_res(llm_model, query, norm_query, retrieved_data)
    state["messages"].append(AIMessage(content=res))
    return {"messages": state["messages"][-1:]}

def memory_retrieval(state: MsgState):
    query = state["messages"][-1].content
    prev_query = state["messages"][-3].content
    prev_res = state["messages"][-2].content
    retrieved_data = state["retrieved_data"]

    res = gen_memory_res(llm_model, query, prev_query, prev_res, retrieved_data)
    state["messages"].append(AIMessage(content=res))
    return {"messages": state["messages"][-1:]}

memory = MemorySaver()

builder = StateGraph(MsgState)

# builder.add_node("llm",llm_call)
builder.add_node(document_retrieval)
builder.add_node(memory_retrieval)
builder.add_node(llm_retrieved_call)
# builder.add_node("router", router)

# builder.add_edge(START, "router")
builder.add_conditional_edges(START, route_condition)
builder.add_edge("document_retrieval", "llm_retrieved_call")
builder.add_edge("llm_retrieved_call", END)
builder.add_edge("memory_retrieval", END)

graph = builder.compile(checkpointer=memory)
# graph = builder.compile()

