from fastapi import FastAPI, status, HTTPException, APIRouter, Request
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

router = APIRouter(
    prefix = "/weaviate",
    tags=["weaviate"]
)

embed_model = OpenAI(
  api_key=os.getenv("EMBED_MODEL_API"),
  base_url="https://integrate.api.nvidia.com/v1"
)

def gen_embedding(prompt):
    response = embed_model.embeddings.create(
        input=prompt,
        model="nvidia/nv-embedqa-e5-v5",
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "NONE"}
    )

    return response.data[0].embedding

@router.get("/.well-known/ready", status_code=status.HTTP_200_OK)
def readiness_check():
    return "Ready"

@router.get('/meta', status_code=status.HTTP_200_OK)
def meta():
    return {'status': 'Ready'}


@router.post("/vectors", status_code=status.HTTP_201_CREATED)
async def vectorize(req: Request):
    try:
        data = await req.json()

        if data is None:
            text_str = (await req.body()).decode("utf-8")
            data = json.loads(text_str)

        if "text" in data:
            if isinstance(data["text"], str):
                texts = [data["text"]]
            elif isinstance(data["text"], list):
                texts = data["text"]
            else:
                texts = data[str(data["text"])]
        else:
            texts = [str(data)]

        embeds = await asyncio.gather(*[asyncio.to_thread(gen_embedding, text) for text in texts])

        if len(texts) == 1:
            return {"vector": embeds[0]}
        else:
            return {"vectors": embeds}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in vectorization: {e}"
        )

