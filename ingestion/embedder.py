"""
SOP Embedder — embeds SOP chunks into ChromaDB using OpenAI embeddings.

Provides ingest, search, and retrieval methods backed by a
persistent ChromaDB collection and OpenAI text-embedding-3-small.
"""

import logging
import os

import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# Load environment variables from .env at module level
load_dotenv()


class SOPEmbedder:
    """Manage SOP chunk embeddings in ChromaDB."""

    def __init__(self, collection_name: str = "sop_chunks"):
        """Initialise embedder with ChromaDB and OpenAI embeddings.

        Args:
            collection_name: Name of the ChromaDB collection to use.

        Raises:
            ValueError: If OPENAI_API_KEY is not set in the environment.
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )

        self.collection_name = collection_name

        # Persistent ChromaDB client — data stored in ./chroma_db
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
        except Exception as exc:
            logger.error("ChromaDB collection error: %s — recreating.", exc)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name
            )

        # OpenAI embeddings model
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )

    # ------------------------------------------------------------------ #
    #  Public methods                                                      #
    # ------------------------------------------------------------------ #

    def ingest(self, chunks: list[dict]) -> int:
        """Embed and store SOP chunks in ChromaDB.

        Clears the existing collection first, then upserts all chunks.

        Args:
            chunks: List of chunk dicts from SOPLoader.load_and_chunk().
                    Each must have: chunk_id, content, section_title, page, step_number.

        Returns:
            Total number of chunks stored.
        """
        # Clear existing data by deleting and recreating the collection
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # Collection may not exist yet
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name
        )

        if not chunks:
            logger.warning("No chunks to ingest.")
            return 0

        ids = []
        embeddings_list = []
        documents = []
        metadatas = []

        for chunk in chunks:
            content = chunk["content"]
            try:
                embedding = self.embeddings.embed_documents([content])[0]
            except Exception as exc:
                logger.error(
                    "⚠️ LLM Error embedding chunk %s: %s. Skipping.",
                    chunk["chunk_id"],
                    exc,
                )
                continue

            ids.append(chunk["chunk_id"])
            embeddings_list.append(embedding)
            documents.append(content)
            metadatas.append({
                "section_title": chunk.get("section_title", ""),
                "page": chunk.get("page", 0),
                "step_number": chunk.get("step_number", 0),
            })

        if ids:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
            )

        stored = self.collection.count()
        logger.info("Ingested %d chunks into ChromaDB.", stored)
        return stored

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Search for the most relevant chunks by semantic similarity.

        Args:
            query: Natural language query string.
            k:     Number of results to return (default 4).

        Returns:
            List of dicts with: content, metadata, distance.
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
        except Exception as exc:
            logger.error("⚠️ LLM Error embedding query: %s", exc)
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s — attempting recovery.", exc)
            # Attempt to recover the collection
            try:
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name
                )
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(k, self.collection.count()),
                )
            except Exception as inner_exc:
                logger.error("Recovery failed: %s", inner_exc)
                return []

        output = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            for doc, meta, dist in zip(docs, metas, dists):
                output.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                })
        return output

    def get_all_chunks(self) -> list[dict]:
        """Return every chunk stored in the collection.

        Returns:
            List of dicts with: id, content, metadata.
        """
        try:
            data = self.collection.get()
        except Exception as exc:
            logger.error("Failed to retrieve all chunks: %s", exc)
            return []

        chunks = []
        if data and data.get("ids"):
            for i, chunk_id in enumerate(data["ids"]):
                chunks.append({
                    "id": chunk_id,
                    "content": data["documents"][i] if data.get("documents") else "",
                    "metadata": data["metadatas"][i] if data.get("metadatas") else {},
                })
        return chunks
