from fastapi import FastAPI, status, Response, HTTPException, Body
from .routers import vectordb
from .ingestion import initialize_weaviate
import os
from dotenv import load_dotenv
from openai import OpenAI
from .utils import gen_single_ip, hybrid_search, gen_final_response
import asyncio

load_dotenv()

app=FastAPI()

client, collection = initialize_weaviate()
print("Weaviate initialized")

llm_model = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("MODEL_API")
)

app.include_router(vectordb.router)

@app.get("/")
async def root():
    return "Hello"

# @app.get("/response")
# async def get_response():
#     res = await asyncio.to_thread(gen_single_ip,llm_model)
#     return res

@app.post("/final",status_code=status.HTTP_201_CREATED)
async def get_response(query:str = Body(...,embed=False)):
    res = await asyncio.to_thread(gen_final_response,llm_model,collection,query)
    return res

@app.post("/response",status_code=status.HTTP_201_CREATED)
async def get_response(query:str = Body(...,embed=False)):
    res = await asyncio.to_thread(gen_single_ip,llm_model,query)
    return res

@app.post("/retrieve",status_code=status.HTTP_201_CREATED)
async def retrieve(query:str = Body(...,embed=False)):
    res = await asyncio.to_thread(hybrid_search,collection,query)
    return res