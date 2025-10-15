import os

# Astra
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")

# HF
HF_API_URL = os.getenv("HF_API_URL", "")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# Reranker
RERANKER_BASE = os.getenv("RERANKER_BASE", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_LOAD_DIR = os.getenv("RERANKER_LOAD_DIR", "")
