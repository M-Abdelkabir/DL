import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=5):
        return self.results[:top_k]

    def format_context(self, results, max_chars=4000):
        return "\n".join(result["text"] for result in results)[:max_chars]


class FakeLLM:
    def generate(self, query, context):
        return f"Réponse pour {query}: {context}"

    def generate_stream(self, query, context):
        yield self.generate(query, context)


fake_retrieval = types.ModuleType("models.retrieval")
fake_retrieval.Retriever = FakeRetriever
fake_retrieval.get_retriever = lambda: FakeRetriever([])
sys.modules["models.retrieval"] = fake_retrieval

fake_llm = types.ModuleType("models.llm")
fake_llm.LLMClient = FakeLLM
fake_llm.get_llm_client = lambda: FakeLLM()
sys.modules["models.llm"] = fake_llm

from models.rag_pipeline import RAGPipeline

sys.modules.pop("models.retrieval", None)
sys.modules.pop("models.llm", None)


class RAGPipelineTest(unittest.TestCase):
    def test_answer_returns_message_when_no_document_found(self):
        rag = RAGPipeline(retriever=FakeRetriever([]), llm=FakeLLM())

        response = rag.answer("question inconnue")

        self.assertEqual(response.sources, [])
        self.assertIn("Aucun document pertinent", response.answer)

    def test_answer_uses_retriever_context_and_returns_sources(self):
        docs = [{"text": "Document FSSM", "source": "source-test", "rank": 1, "score": 0.9}]
        rag = RAGPipeline(retriever=FakeRetriever(docs), llm=FakeLLM())

        response = rag.answer("inscription", top_k=1)

        self.assertEqual(response.sources, docs)
        self.assertIn("Document FSSM", response.answer)


if __name__ == "__main__":
    unittest.main()
