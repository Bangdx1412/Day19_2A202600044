"""Project configuration.

Secrets are read from environment variables. Do not hardcode API keys or
database passwords in this file.
"""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"


def load_dotenv(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE pairs from .env without requiring extra packages."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_dotenv()

DATA_DIR = BASE_DIR / "data"
EVALUATION_DIR = BASE_DIR / "evaluation"
OUTPUT_DIR = BASE_DIR / "outputs"

CORPUS_PATH = DATA_DIR / "tech_corpus.txt"
TRIPLES_PATH = DATA_DIR / "triples.json"
NODE_EMBEDDINGS_PATH = DATA_DIR / "node_embeddings.json"
GRAPH_IMAGE_PATH = OUTPUT_DIR / "knowledge_graph.png"

# OpenAI is optional. The lab can run fully offline with deterministic fallback
# extraction and answer synthesis.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
USE_OPENAI = bool(OPENAI_API_KEY)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "64"))

# Neo4j is optional. The default pipeline uses a local NetworkX graph so the
# deliverables can be generated without a running database.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

DEFAULT_GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "networkx").lower()
MAX_HOPS = int(os.getenv("MAX_HOPS", "2"))

FLAT_RAG_TOP_K = int(os.getenv("FLAT_RAG_TOP_K", "3"))

# Approximate generation prices used only for the benchmark report.
# Defaults are conservative placeholders for local/offline runs.
INPUT_COST_PER_1K_TOKENS = float(os.getenv("INPUT_COST_PER_1K_TOKENS", "0.00015"))
OUTPUT_COST_PER_1K_TOKENS = float(os.getenv("OUTPUT_COST_PER_1K_TOKENS", "0.00060"))
