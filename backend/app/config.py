from pydantic_settings import BaseSettings
from pydantic import BaseModel

class EmbeddingSettings(BaseModel):
    em_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    em_model_kwargs: dict = {'device': 'cpu'}
    em_encode_kwargs: dict = {'normalize_embeddings': True} 

class Settings(BaseSettings):
    qdrant_host: str = 'localhost'
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = 'qa_collection'
    embedding_settings: EmbeddingSettings = EmbeddingSettings()
    text_splitter: dict = { "chunk_size": 100, "chunk_overlap": 20}

    class Config:
        env_file = ".env"


settings = Settings()