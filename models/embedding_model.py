"""
Embedding model wrapper.
Compatible avec les embeddings précomputés (dim=768, float32).
Modèle par défaut : paraphrase-multilingual-mpnet-base-v2 (768 dims, multilingue FR/AR/EN)
"""

import os

# Dimension attendue selon embeddings.npy
EMBEDDING_DIM = 768
DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"


class EmbeddingModel:
    """Wrapper autour de SentenceTransformer pour encoder des textes en vecteurs."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def load(self):
        """Charge le modèle (lazy loading)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"[EmbeddingModel] Chargement du modèle : {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != EMBEDDING_DIM:
                raise ValueError(
                    f"Dimension du modèle ({actual_dim}) incompatible "
                    f"avec les embeddings existants ({EMBEDDING_DIM}). "
                    f"Utilisez un modèle à {EMBEDDING_DIM} dimensions."
                )
        return self

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def encode(self, texts: list[str] | str, batch_size: int = 32, normalize: bool = True):
        """
        Encode une liste de textes en vecteurs numpy.

        Args:
            texts: texte ou liste de textes à encoder
            batch_size: taille des batchs (défaut 32)
            normalize: normalisation L2 pour similarité cosinus (défaut True)

        Returns:
            np.ndarray de shape (n, 768), dtype float32
        """
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str):
        """Encode une seule requête utilisateur. Retourne un vecteur (768,)."""
        return self.encode([query])[0]

    def get_dim(self) -> int:
        return EMBEDDING_DIM


# Instance partagée (singleton)
_embedding_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    """Retourne l'instance singleton du modèle d'embedding."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
