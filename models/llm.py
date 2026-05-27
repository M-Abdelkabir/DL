"""
llm.py — LLM wrapper utilisant Groq API
Modèle par défaut : llama-3.3-70b-versatile (rapide, multilingue FR/AR/EN)
"""

import os
from groq import Groq

# ── Configuration ──────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS", "1024"))
TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0.2"))

SYSTEM_PROMPT = """Tu es un assistant expert de la Faculté des Sciences Semlalia (FSSM) 
de l'Université Cadi Ayyad (UCA) à Marrakech.
Réponds uniquement à partir du contexte fourni.
Si la réponse ne se trouve pas dans le contexte, dis-le clairement et poliment.
Réponds en français de façon précise et concise."""


class LLMClient:
    """Client Groq pour la génération de réponses RAG."""

    def __init__(
        self,
        model: str         = DEFAULT_MODEL,
        api_key: str       = GROQ_API_KEY,
        temperature: float = TEMPERATURE,
        max_tokens: int    = MAX_TOKENS,
    ):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY est manquante. Définissez cette variable d'environnement "
                "avant de lancer l'application."
            )

        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.client      = Groq(api_key=api_key)

    def generate(self, query: str, context: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        Génère une réponse à partir de la question et du contexte récupéré.

        Args:
            query: question de l'utilisateur
            context: texte de contexte formaté (sortie de Retriever.format_context)
            system_prompt: instruction système

        Returns:
            Réponse textuelle du LLM
        """
        user_message = f"""Contexte :
{context}

Question : {query}

Réponds uniquement à partir du contexte ci-dessus."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, query: str, context: str, system_prompt: str = SYSTEM_PROMPT):
        """
        Version streaming — yield les tokens au fur et à mesure.

        Usage :
            for token in llm.generate_stream(query, context):
                print(token, end="", flush=True)
        """
        user_message = f"Contexte :\n{context}\n\nQuestion : {query}"

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Singleton ──────────────────────────────────────────────────────────────
_llm_client: LLMClient | None = None

def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
