# ==============================================
# Astra DB Storage Utilities for CodeRank
# ==============================================

from dotenv import load_dotenv
load_dotenv()  # Load environment variables early

import time
from typing import Dict, Any, List
from astrapy import DataAPIClient

# --- Robust Exception Imports (Handles all astrapy versions) ---
try:
    from astrapy.exceptions import DataAPIException
except ImportError:
    try:
        from astrapy.exceptions import DataAPIError as DataAPIException
    except ImportError:
        class DataAPIException(Exception):
            """Fallback if astrapy exception is unavailable."""
            pass

# --- Project Settings ---
from coderank_lc.core.settings import (
    ASTRA_DB_APPLICATION_TOKEN,
    ASTRA_DB_API_ENDPOINT,
    ASTRA_DB_KEYSPACE,
)

# --- AstraDB Client Setup ---
client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
db = client.get_database(ASTRA_DB_API_ENDPOINT)


# ==============================================
# Utility: Ensure a Collection Exists
# ==============================================
def ensure_collection_exists(name: str):
    """Ensure a collection exists in Astra DB and return the collection object."""
    try:
        db.create_collection(name, keyspace=ASTRA_DB_KEYSPACE)
        print(f"✅ Created new collection: {name}")
    except DataAPIException as e:
        if "already exists" in str(e).lower():
            print(f"ℹ️ Collection '{name}' already exists.")
        else:
            print(f"⚠️ Error creating collection '{name}': {e}")

    # Wait briefly for Astra to confirm creation
    for _ in range(5):
        try:
            if name in db.list_collection_names():
                return db.get_collection(name, keyspace=ASTRA_DB_KEYSPACE)
            time.sleep(2)
        except Exception:
            time.sleep(2)

    raise RuntimeError(f"❌ Could not verify existence of collection: {name}")


# ==============================================
# Initialize Core Collections
# ==============================================
responses = ensure_collection_exists("responses")
feedback = ensure_collection_exists("feedback")


# ==============================================
# CRUD Operations
# ==============================================
def store_response(doc: Dict[str, Any]) -> str:
    """Insert an agent response into Astra DB."""
    try:
        res = responses.insert_one(doc)
        print(f"📥 Stored response from {doc.get('agent')}")
        return str(getattr(res, "inserted_id", res))
    except Exception as e:
        print(f"⚠️ Error storing response: {e}")
        return ""


def store_feedback(doc: Dict[str, Any]) -> str:
    """Insert a human feedback entry."""
    try:
        res = feedback.insert_one(doc)
        print(f"📝 Stored feedback preference: {doc.get('preferred')}")
        return str(getattr(res, "inserted_id", res))
    except Exception as e:
        print(f"⚠️ Error storing feedback: {e}")
        return ""


def store_reranker_score(doc: Dict[str, Any]) -> str:
    """Insert a reranker score for offline fine-tuning."""
    coll_name = "reranker_scores"
    try:
        coll = ensure_collection_exists(coll_name)
        res = coll.insert_one(doc)
        print(f"🏁 Stored reranker score for {doc.get('agent')} → {doc.get('score')}")
        return str(getattr(res, "inserted_id", res))
    except Exception as e:
        print(f"⚠️ Error storing reranker score: {e}")
        return ""


def list_recent_feedback(limit: int = 1000) -> List[Dict[str, Any]]:
    """Fetch the most recent feedback documents."""
    try:
        # Prefer sorting by timestamp if available
        try:
            docs = feedback.find({}, sort={"created_at": -1}, limit=limit)
        except Exception:
            # Fallback: no sort
            docs = feedback.find({}, limit=limit)
        return list(docs)
    except Exception as e:
        print(f"⚠️ Error listing feedback: {e}")
        return []

