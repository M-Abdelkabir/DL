import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeGroqClient:
    def __init__(self, api_key):
        self.api_key = api_key


fake_groq = types.ModuleType("groq")
fake_groq.Groq = FakeGroqClient
sys.modules["groq"] = fake_groq

from models.llm import LLMClient


class LLMClientTest(unittest.TestCase):
    def test_llm_requires_api_key(self):
        with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
            LLMClient(api_key=None)

    def test_llm_accepts_api_key(self):
        client = LLMClient(api_key="test-key")

        self.assertEqual(client.client.api_key, "test-key")


if __name__ == "__main__":
    unittest.main()
