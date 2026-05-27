import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.SentenceTransformer = object
sys.modules["sentence_transformers"] = fake_sentence_transformers

fake_chromadb = types.ModuleType("chromadb")
fake_chromadb.PersistentClient = object
sys.modules["chromadb"] = fake_chromadb

fake_chromadb_config = types.ModuleType("chromadb.config")
fake_chromadb_config.Settings = object
sys.modules["chromadb.config"] = fake_chromadb_config

from models.retrieval import normalize_metadata


class SearchHelpersTest(unittest.TestCase):
    def test_normalize_metadata_flattens_nested_metadata(self):
        item = {
            "id": "doc_001",
            "text": "contenu",
            "metadata": {
                "source": "FSSM",
                "tags": ["inscription", "admin"],
            },
        }

        meta = normalize_metadata(item)

        self.assertEqual(meta["id"], "doc_001")
        self.assertEqual(meta["source"], "FSSM")
        self.assertIn("inscription", meta["tags"])
        self.assertNotIn("text", meta)


if __name__ == "__main__":
    unittest.main()
