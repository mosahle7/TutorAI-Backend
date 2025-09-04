from fastapi import FastAPI, status, HTTPException, APIRouter, Request
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import asyncio
# from FlagEmbedding import FlagReranker
import requests

load_dotenv()

# reranker = FlagReranker('BAAI/bge-reranker-base', 
#                         cache_dir='.models/', 
#                         use_fp16=True)

router = APIRouter(
    prefix = "/weaviate",
    tags=["weaviate"]
)

#nv-embedqa-e5-v5
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

@router.post("/rerank", status_code=status.HTTP_200_OK)
async def rerank(req: Request):
    """
    Expected JSON from Weaviate:
    {
        "query": "some question",
        "documents": ["doc1 text", "doc2 text", ...]
    }
    """
    try:
        data = await req.json()
        query = data.get("query")
        documents = data.get("documents")
        print(documents)

        if not isinstance(query, str) or not isinstance(documents, list):
            raise HTTPException(status_code=400, detail="Invalid input format")

        if not documents:
            return {"scores": []}

        headers = {
            "Authorization": f"Bearer {os.getenv('RERANKER_MODEL_API')}",
            "Accept": "application/json",
        }

        payload = {
            "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
            "query": {"text": query},
            "passages" :[{"text": doc["text"] if isinstance(doc,dict) else doc} for doc in documents]
        }

        resp = requests.post(
            url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
            headers=headers,
            json=payload,
            timeout=60
        )

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"NVIDIA API error: {resp.text}")
        
        result = resp.json()

        # Prepare pairs for FlagReranker
        # pairs = [(query, doc) for doc in documents]
        # scores = reranker.compute_score(pairs)

        # scores_list = scores.tolist() if hasattr(scores, "tolist") else scores

        # print("Rerank done!")
        # return {
        #     "scores": [
        #         {"document": doc, "score": float(score)}
        #         for doc, score in zip(documents, scores_list)
        #     ]
        # }

        scores = []
        for item in result.get("rankings", []):
            doc = documents[item["index"]]  # map index back to document
            scores.append({
                "document": doc,
                "score": item.get("logit", 0.0)
            })

        print("Rerank done via NVIDIA API!")
        return {"scores": scores}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in reranking: {e}"
        )