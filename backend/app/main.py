from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Request
from app.api.api_v1.routers.rag import qa_router
from config import settings
import typing as t
import uvicorn



# from qdrant_engine import QdrantIndex
# from sentence_transformers import SentenceTransformer


app = FastAPI(
    title="AI Starter Kit backend API", docs_url="/docs"
)


origins = [
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root(request: Request):
    return {"message": "Server is up and running!"}


app.include_router(qa_router, prefix="/api/v1", tags=["QA"])


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=8000)