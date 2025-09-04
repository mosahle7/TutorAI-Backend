from fastapi import FastAPI, status, Response, HTTPException, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from .ingestion import initialize_client, initialize_collection
import os
from dotenv import load_dotenv
from openai import OpenAI
from .utils import gen_single_ip, hybrid_search, gen_final_response, check_mode, list_files
import asyncio
import re

load_dotenv()

app=FastAPI()

save_dir = "/root/TutorAI/backend/app/data"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = initialize_client()
print("Weaviate initialized")

collection, terms = initialize_collection(client)

llm_model = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("MODEL_API")
)

# app.include_router(vectordb.router)

@app.get("/")
async def root():
    return "Hello"

# @app.get("/response")
# async def get_response():
#     res = await asyncio.to_thread(gen_single_ip,llm_model)
#     return res

@app.options("/final")
async def options_final():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/final",status_code=status.HTTP_201_CREATED)
async def get_response(query:str = Body(...,embed=False)):
    def generate():
        return gen_final_response(llm_model,collection,query,terms)

    
    return StreamingResponse(
        generate(),
        media_type = "text/plain",
        headers = {
            "Cache-control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",              
            "Access-Control-Allow-Methods": "POST, OPTIONS", 
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/response",status_code=status.HTTP_201_CREATED)
async def get_response(query:str = Body(...,embed=False)):
    res = await asyncio.to_thread(gen_single_ip,llm_model,query)
    return res

@app.post("/mode",status_code=status.HTTP_201_CREATED)
async def get_response(query:str = Body(...,embed=False)):
    res = await asyncio.to_thread(check_mode,llm_model,query)
    return res

@app.post("/retrieve",status_code=status.HTTP_201_CREATED)
async def retrieve(query:str = Body(...,embed=False)):
    res = await asyncio.to_thread(hybrid_search,collection,query)
    return res

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global collection, terms
    os.makedirs(save_dir, exist_ok=True)
    file.filename = re.sub(r"[^A-Za-z0-9_]+","_",file.filename)
    file_path = os.path.join(save_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    collection, terms = initialize_collection(client)
    return {"filename": file.filename, "message": "File uploaded successfully"}

@app.get("/list_docs")
def list_docs():
    files = list_files()
    return files


@app.get("/select_collection")
def select_collection(collection_name: str):
    global collection
    # try:
    collection = client.collections.get(collection_name)
    # except Exception as e:
    #     print(f"Error selecting collection: {e}")
    return {"collection_name":collection_name, "collection": collection.name}
    
@app.get("/show_collection")
def show_collection():
    global collection
    return collection.name