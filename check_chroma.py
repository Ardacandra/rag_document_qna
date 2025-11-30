#!/usr/bin/env python3
"""
Inspect and sample contents stored in a ChromaDB persistent directory.

Example:
  python check_chroma.py --path chroma_db --collection advanced_rag_collection --sample 3

The script prints collections, collection counts, and a small sample of stored
documents and metadata to help debug indexing/persistence issues.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import chromadb
except Exception as e:
    print("Error: chromadb is not installed. Install with 'pip install chromadb'.")
    print(e)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect a ChromaDB persistent folder and sample collection contents.")
    p.add_argument("--path", default="chroma_db", help="Path to chroma DB folder (PersistentClient path)")
    p.add_argument("--collection", default=None, help="Collection name to inspect (if omitted, list collections) ")
    p.add_argument("--sample", type=int, default=3, help="Number of sample documents to print from the collection")
    return p.parse_args()


def list_collections(client) -> list:
    """Return a list of collection names supported by this client.

    Different chromadb versions expose different client APIs; try a few options.
    """
    # Try the documented helper
    try:
        cols = client.list_collections()
        # list_collections often returns objects with 'name' field
        names = [c.get("name") if isinstance(c, dict) else getattr(c, "name", None) for c in cols]
        return [n for n in names if n]
    except Exception:
        pass

    # Fallback: try get_or_create_collection on a well-known name to probe
    # (not ideal, but safe because it won't remove existing collections)
    try:
        # Some clients expose 'get_or_create_collection' only
        # We can't enumerate without a direct API; return empty and let callers try to open a collection
        return []
    except Exception:
        return []


def inspect_collection(client, collection_name: str, sample: int = 3) -> None:
    print(f"Inspecting collection: {collection_name}")
    try:
        coll = client.get_or_create_collection(collection_name)
    except Exception as e:
        print(f"Failed to get or create collection '{collection_name}': {e}")
        return

    # Count
    try:
        count = coll.count()
    except Exception:
        # Some versions use len(coll.get(...))
        try:
            all_docs = coll.get(include=["ids"]) if hasattr(coll, "get") else {}
            count = len(all_docs.get("ids", []))
        except Exception:
            count = None

    print(f"Collection item count: {count}")

    # Try fetching a small sample of documents using several include variants
    includes_variants = [
        ["documents", "metadatas", "embeddings"],
        ["documents", "metadatas"],
        [],
    ]
    result = None
    last_err = None
    for includes in includes_variants:
        try:
            if includes:
                result = coll.get(include=includes, limit=sample)
            else:
                result = coll.get(limit=sample)
            break
        except TypeError:
            # Some chromadb versions expect an 'ids' kwarg; try that signature
            try:
                if includes:
                    result = coll.get(ids=None, include=includes, limit=sample)
                else:
                    result = coll.get(ids=None, limit=sample)
                break
            except Exception as e:
                last_err = e
                continue
        except Exception as e:
            last_err = e
            # If the error mentions 'ids' in include, try without 'ids'
            if "Expected include item" in str(e) or "got ids" in str(e):
                continue
            continue

    if result is None:
        print(f"Failed to retrieve documents from collection: {last_err}")
        return

    # Normalize results - chromadb versions differ in returned keys
    ids = result.get("ids") or result.get("id") or []
    docs = result.get("documents") or result.get("document") or []
    metas = result.get("metadatas") or result.get("metadata") or []

    n = max(len(ids), len(docs), len(metas))
    if n == 0:
        print("No documents found in collection.")
        return

    print(f"Showing up to {sample} sample items from the collection:")
    for i in range(min(sample, n)):
        print("---")
        print(f"ID: {ids[i] if i < len(ids) else 'N/A'}")
        meta = metas[i] if i < len(metas) else None
        if meta:
            try:
                print("Metadata:")
                print(json.dumps(meta, indent=2, ensure_ascii=False))
            except Exception:
                print(repr(meta))
        doc = docs[i] if i < len(docs) else None
        if doc:
            # Print a short preview
            preview = (doc[:1000] + "...") if len(doc) > 1000 else doc
            print("Document preview:")
            print(preview.replace("\n", " "))


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        print(f"Chroma DB path '{path}' does not exist.")
        sys.exit(1)

    try:
        client = chromadb.PersistentClient(path=str(path))
    except Exception as e:
        print(f"Failed to open chromadb PersistentClient at {path}: {e}")
        sys.exit(1)

    # If collection not provided, try to list collections (best-effort)
    if not args.collection:
        cols = list_collections(client)
        if cols:
            print("Collections found:")
            for c in cols:
                print(f" - {c}")
        else:
            print("No collection list API available; please pass --collection <name> to inspect a specific collection.")
        sys.exit(0)

    inspect_collection(client, args.collection, sample=args.sample)


if __name__ == "__main__":
    main()
