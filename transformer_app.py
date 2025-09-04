from fastapi import FastAPI
from app.routers.vectordb import router

# Dedicated transformer service
transformer_app = FastAPI(title="Weaviate Transformer Service")
transformer_app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(transformer_app, host="0.0.0.0", port=8000)