from fastapi import FastAPI, status, Response, HTTPException, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from .ingestion import initialize_client, initialize_collection
import os
from dotenv import load_dotenv
from openai import OpenAI
from .utils import gen_single_ip, hybrid_search, gen_final_response, check_mode, list_files, is_pdf
from .agent import graph, MsgState, config
from .nodes.doc_retrieve import check_mode
import asyncio
import re
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from .config import client, collection, terms
from . import vdb_config
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

app=FastAPI()

save_dir = "./app/data"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# client = initialize_client()
# print("Weaviate initialized")

# collection, terms = initialize_collection(client)

# llm_model = OpenAI(
#   base_url = "https://integrate.api.nvidia.com/v1",
#   api_key = os.getenv("MODEL_API")
# )

llm_model = ChatNVIDIA(
    base_url="https://integrate.api.nvidia.com/v1",
    # api_key=os.getenv("MODEL_API"),
    # model="nvidia/llama-3.1-nemotron-nano-8b-v1",
    api_key=os.getenv("LLAMA_3.170B"),
    model="meta/llama-3.1-70b-instruct",
    temperature=0,
    top_p=0.75,
    max_retries=3,
    max_completion_tokens=200
)

# app.include_router(vectordb.router)

@app.get("/")
async def root():
    
    return "Hello"

# @app.get("/response")
# async def get_response():
#     res = await asyncio.to_thread(gen_single_ip,llm_model)
#     return res

# @app.options("/final")
# async def options_final():
#     return Response(
#         status_code=200,
#         headers={
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Methods": "POST, OPTIONS",
#             "Access-Control-Allow-Headers": "*",
#         }
#     )

# @app.post("/final",status_code=status.HTTP_201_CREATED)
# async def get_response(query:str = Body(...,embed=False)):
#     def generate():
#         return gen_final_response(llm_model,collection,query,terms)

    
#     return StreamingResponse(
#         generate(),
#         media_type = "text/plain",
#         headers = {
#             "Cache-control": "no-cache",
#             "Connection": "keep-alive",
#             "Access-Control-Allow-Origin": "*",              
#             "Access-Control-Allow-Methods": "POST, OPTIONS", 
#             "Access-Control-Allow-Headers": "*",
#         }
#     )

# def gen_agentic(query: str):
#     state = graph.get_state(config=config)

#     if state.values == {}:
#         state.values["messages"] = [HumanMessage(content=query)]
#         state.values["retrieved_data"]=""
#         print("State initialized")
#     else:
#         state.values["messages"].append(HumanMessage(content=query))

#     res = graph.invoke(state.values, config=config)

#     return res["messages"][-1].content

# @app.post("/agentic_final",status_code=status.HTTP_201_CREATED)
# async def get_agentic(query:str = Body(...,embed=False)):
#     res = await asyncio.to_thread(gen_agentic, query)
#     return res

def gen_agentic_stream(query: str):
    print(f"🔍 Starting gen_agentic_stream with query: {query}")
    mode = check_mode(llm_model, query)
    state = graph.get_state(config=config)

    inputs = {
            "messages": [HumanMessage(content=query)],
            "retrieved_data": "",
            "mode": mode
        }
    
    if state.values == {}:
        state.values.update(inputs)
        print("🆕 State initialized")

    else:
        state.values["messages"].append(HumanMessage(content=query))
        state.values["retrieved_data"] = state.values.get("retrieved_data", "")
        state.values["mode"] = mode
        print("📝 Added to existing state")
    
    print("🚀 Starting graph stream...")
    
    chunk_count = 0
    for message, metadata in graph.stream(inputs, config=config, stream_mode="messages"):
        if metadata.get("langgraph_node") == "llm_retrieved_call":
            if isinstance(message, AIMessage):
                content = message.content
                if content:  # Only yield non-empty content
                    chunk_count += 1
                    yield content
        elif metadata.get("langgraph_node") == "memory_retrieval":
             if isinstance(message, AIMessage):
                content = message.content
                if content:  # Only yield non-empty content
                    chunk_count += 1
                    yield content
    

    print(f"🏁 Finished streaming. Total chunks: {chunk_count}")


@app.post("/agent_stream",status_code=status.HTTP_201_CREATED)
async def get_agentic(query:str = Body(...,embed=False)):
    async def event_generator():
        for chunk in gen_agentic_stream(query):
            yield chunk
            await asyncio.sleep(0)

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",              
            "Access-Control-Allow-Methods": "POST, OPTIONS", 
            "Access-Control-Allow-Headers": "*",
        },
    )


# @app.post("/response",status_code=status.HTTP_201_CREATED)
# async def get_response(query:str = Body(...,embed=False)):
#     res = await asyncio.to_thread(gen_single_ip,llm_model,query)
#     return res


# @app.post("/mode",status_code=status.HTTP_201_CREATED)
# async def get_response(query:str = Body(...,embed=False)):
#     res = await asyncio.to_thread(check_mode,llm_model,query)
#     return res

# @app.post("/retrieve",status_code=status.HTTP_201_CREATED)
# async def retrieve(query:str = Body(...,embed=False)):
#     res = await asyncio.to_thread(hybrid_search,collection,query)
#     return res

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # global collection, terms
    os.makedirs(save_dir, exist_ok=True)

    base, ext = os.path.splitext(file.filename)
    safe_base = re.sub(r"[^A-Za-z0-9_]+","_",base)

    safe_filename = safe_base + ext
    file_path = os.path.join(save_dir, safe_filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    vdb_config.collection, vdb_config.terms = initialize_collection(vdb_config.client)
    return {"filename": safe_filename, "message": "File uploaded successfully"}

@app.get("/list_docs")
def list_docs():
    files = list_files()
    return files


@app.get("/select_collection")
def select_collection(collection_name: str):
    # global collection
    # try:
    vdb_config.collection = vdb_config.client.collections.get(collection_name)
    # except Exception as e:
    #     print(f"Error selecting collection: {e}")
    return {"collection_name":collection_name, "collection": vdb_config.collection.name}

@app.get("/show_collection")
def show_collection():
    return vdb_config.collection.name

@app.get("/read_docs/{filename}")
async def read_doc(filename: str):
    path = os.path.join(save_dir, filename)
    if is_pdf(path):
        return FileResponse(
            path, media_type='application/pdf', 
            filename=filename,
            headers={"Content-Disposition": f'inline; filename="{filename}"'})
    else:
        return FileResponse(
            path, 
            media_type="text/plain",
            filename=filename,
            headers={"Content-Disposition": f'inline; filename="{filename}"'})

@app.get("/download_docs/{filename}")
async def download_doc(filename:str):
    path = os.path.join(save_dir, filename)
    return FileResponse(
        path,
        filename = filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
