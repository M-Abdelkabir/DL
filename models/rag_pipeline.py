"""
Pipeline RAG complet.
Orchestre : EmbeddingModel → Retriever (FAISS) → LLMClient → réponse.

Usage rapide :
    from models.rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    print(rag.answer("Quelles sont les conditions d'inscription ?"))
"""

from dataclasses import dataclass, field
from models.retrieval import get_retriever, Retriever
from models.llm import get_llm_client, LLMClient


@dataclass
class RAGResponse:
    """Réponse structurée du pipeline RAG."""
    query: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    top_k: int = 5

    def __str__(self):
        src_lines = "\n".join(
            f"  [{r['rank']}] {r.get('source', '?')} (score: {r.get('score', 0):.3f})"
            for r in self.sources
        )
        return f"Question : {self.query}\n\nRéponse :\n{self.answer}\n\nSources :\n{src_lines}"


class RAGPipeline:
    """
    Pipeline RAG (Retrieval-Augmented Generation).

    Étapes :
      1. Encode la requête en vecteur (EmbeddingModel)
      2. Recherche les documents pertinents (Retriever + FAISS)
      3. Formate le contexte
      4. Génère la réponse (LLM)
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
        top_k: int = 5,
        max_context_chars: int = 4000,
    ):
        self.retriever = retriever or get_retriever()
        self.llm = llm or get_llm_client()
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    def answer(self, query: str, top_k: int | None = None) -> RAGResponse:
        """
        Répond à une question en mode complet (bloquant).

        Args:
            query: question de l'utilisateur
            top_k: surcharge le nombre de documents à récupérer

        Returns:
            RAGResponse avec la réponse et les sources
        """
        k = top_k or self.top_k

        # 1. Retrieval
        results = self.retriever.search(query, top_k=k)

        if not results:
            return RAGResponse(
                query=query,
                answer="Aucun document pertinent trouvé pour répondre à cette question.",
                sources=[],
                top_k=k,
            )

        # 2. Format contexte
        context = self.retriever.format_context(results, max_chars=self.max_context_chars)

        # 3. Génération
        answer_text = self.llm.generate(query=query, context=context)

        return RAGResponse(query=query, answer=answer_text, sources=results, top_k=k)

    def answer_stream(self, query: str, top_k: int | None = None):
        """
        Version streaming — yield les tokens au fur et à mesure.

        Usage :
            for token in rag.answer_stream("Ma question"):
                print(token, end="", flush=True)
        """
        k = top_k or self.top_k
        results = self.retriever.search(query, top_k=k)

        if not results:
            yield "Aucun document pertinent trouvé."
            return

        context = self.retriever.format_context(results, max_chars=self.max_context_chars)

        yield from self.llm.generate_stream(query=query, context=context)

    def get_sources(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retourne uniquement les sources sans générer de réponse."""
        return self.retriever.search(query, top_k=top_k or self.top_k)


# Singleton
_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """Retourne l'instance singleton du pipeline RAG."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
