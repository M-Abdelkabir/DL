"""
retrieval.py — Recherche sémantique tout-en-un avec ChromaDB
=============================================================
Contient :
  - EmbeddingModel  : encode les requêtes (SentenceTransformer 768d)
  - Retriever       : recherche dans ChromaDB
  - build_chroma()  : migration embeddings.npy + metadata.json → ChromaDB (1 seule fois)

Migration (une seule fois) :
    python models/retrieval.py

Usage dans rag_pipeline.py :
    from models.retrieval import get_retriever
    retriever = get_retriever()
    results = retriever.search("Quelles sont les filières ?", top_k=5)
"""

import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR         = Path(__file__).parent.parent
EMBEDDINGS_PATH  = ROOT_DIR / "embeddings" / "embeddings.npy"
METADATA_PATH    = ROOT_DIR / "embeddings" / "metadata.json"
CHROMA_DIR       = ROOT_DIR / "chroma_db"
COLLECTION_NAME  = "fssm_docs"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"  # 768 dims FR/AR/EN
EMBEDDING_DIM        = 768
BATCH_SIZE           = 50


def normalize_metadata(item: dict) -> dict:
    """Retourne des métadonnées plates compatibles avec ChromaDB."""
    meta = dict(item.get("metadata") or {})
    for key, value in item.items():
        if key not in ("text", "content", "metadata"):
            meta.setdefault(key, value)
    return {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
            for k, v in meta.items()}


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingModel:
    """Encode des textes en vecteurs float32 (dim=768) — lazy loading."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("[Embedding] Chargement du modèle '%s' …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            dim = self._model.get_sentence_embedding_dimension()
            if dim != EMBEDDING_DIM:
                raise ValueError(
                    f"Modèle dim={dim} incompatible avec embeddings existants (dim={EMBEDDING_DIM})"
                )
            log.info("[Embedding] Modèle prêt (dim=%d) ✓", dim)

    @property
    def model(self):
        self._load()
        return self._model

    def encode(self, texts: list[str] | str, normalize: bool = True):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        ).astype(np.float32)

    def encode_query(self, query: str):
        """Retourne un vecteur (768,) pour une requête utilisateur."""
        return self.encode([query])[0]


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVER — ChromaDB
# ══════════════════════════════════════════════════════════════════════════════

class Retriever:
    """Recherche sémantique dans la collection ChromaDB persistante."""

    def __init__(
        self,
        chroma_dir: str      = str(CHROMA_DIR),
        collection_name: str = COLLECTION_NAME,
    ):
        self.chroma_dir      = chroma_dir
        self.collection_name = collection_name
        self._collection     = None
        self._emb            = EmbeddingModel()

    def _load(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings

            log.info("[Retriever] Connexion ChromaDB dans '%s' …", self.chroma_dir)
            client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            n = self._collection.count()
            log.info("[Retriever] %d documents dans '%s'", n, self.collection_name)

            if n == 0:
                log.warning("[Retriever] Collection vide — lancement automatique de build_chroma() …")
                build_chroma()
                # Reconnexion après build
                self._collection = client.get_collection(name=self.collection_name)
                log.info("[Retriever] %d documents indexés après build ✓", self._collection.count())

    @property
    def collection(self):
        self._load()
        return self._collection

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retourne les top_k chunks les plus proches de la requête.

        Chaque résultat contient :
          - text   (str)   : contenu du chunk
          - score  (float) : similarité cosinus [0, 1]
          - rank   (int)   : rang (1 = meilleur)
          - +tous les champs de metadata.json
        """
        query_vec = self._emb.encode_query(query).tolist()
        count = self.collection.count()
        if count == 0:
            log.error('[Retriever] Aucun document indexé — vérifie embeddings/metadata.json')
            return []
        n_results = min(top_k, count)

        res = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for rank, (doc, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ):
            entry          = dict(meta)
            entry["text"]  = doc
            entry["score"] = round(1.0 - dist, 4)  # distance cosinus → similarité
            entry["rank"]  = rank + 1
            output.append(entry)

        return output

    def format_context(self, results: list[dict], max_chars: int = 4000) -> str:
        """Formate les résultats en bloc texte prêt pour le LLM."""
        parts, total = [], 0
        for r in results:
            text   = r.get("text", r.get("content", ""))
            source = r.get("source", r.get("filename", "inconnu"))
            block  = f"[Source: {source} | Score: {r.get('score', 0):.3f}]\n{text}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n---\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# MIGRATION : embeddings.npy + metadata.json → ChromaDB
# Lance : python models/retrieval.py
# ══════════════════════════════════════════════════════════════════════════════

def build_chroma():
    import numpy as np
    import chromadb
    from chromadb.config import Settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    for path in (EMBEDDINGS_PATH, METADATA_PATH):
        if not path.exists():
            log.error("Fichier introuvable : %s", path)
            sys.exit(1)

    log.info("Chargement embeddings.npy …")
    embeddings = np.load(str(EMBEDDINGS_PATH)).astype(np.float32)

    log.info("Chargement metadata.json …")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata: list[dict] = json.load(f)

    n = len(metadata)
    assert embeddings.shape[0] == n, f"Mismatch : {embeddings.shape[0]} embeddings vs {n} metadata"
    log.info("%d documents à indexer (dim=%d)", n, embeddings.shape[1])

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Recrée la collection proprement
    try:
        client.delete_collection(COLLECTION_NAME)
        log.info("Ancienne collection supprimée.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, n, BATCH_SIZE):
        end   = min(start + BATCH_SIZE, n)
        batch = metadata[start:end]

        # Nettoie les métadonnées (ChromaDB n'accepte que des strings)
        clean_meta = [normalize_metadata(m) for m in batch]

        collection.add(
            ids        = [str(i) for i in range(start, end)],
            embeddings = embeddings[start:end].tolist(),
            documents  = [m.get("text", m.get("content", "")) for m in batch],
            metadatas  = clean_meta,
        )
        log.info("  [%d/%d] insérés", end, n)

    log.info("✓ ChromaDB prêt — %d documents dans '%s'", collection.count(), COLLECTION_NAME)
    log.info("  Dossier : %s", CHROMA_DIR.resolve())


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_retriever: Retriever | None = None

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — migration
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    build_chroma()
