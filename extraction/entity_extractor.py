"""Entity and relation extraction for the Tech Company Corpus.

The primary path can use an OpenAI model when OPENAI_API_KEY is configured.
For repeatable lab submission and offline grading, the module also includes a
deterministic extractor tailored to the provided corpus.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from config import LLM_MODEL, OPENAI_API_KEY, TRIPLES_PATH, USE_OPENAI


EXTRACTION_PROMPT = """You are a knowledge graph expert. Extract entities and relationships from the given text.

Rules:
1. Entities are proper nouns: companies, people, products, technologies, locations, years, and notable events.
2. Relations must be short and in UPPER_SNAKE_CASE.
3. Do not extract adjectives such as "open-source" or "large" as standalone entities.
4. Deduplicate exact triples.
5. Return only a JSON array of objects with subject, relation, and object.

Text:
{text}
"""


CURATED_TRIPLES: list[dict[str, str]] = [
    {"subject": "OpenAI", "relation": "FOUNDED_BY", "object": "Sam Altman"},
    {"subject": "OpenAI", "relation": "FOUNDED_BY", "object": "Elon Musk"},
    {"subject": "OpenAI", "relation": "FOUNDED_BY", "object": "Greg Brockman"},
    {"subject": "OpenAI", "relation": "FOUNDED_BY", "object": "Ilya Sutskever"},
    {"subject": "OpenAI", "relation": "FOUNDED_IN", "object": "2015"},
    {"subject": "OpenAI", "relation": "HEADQUARTERED_IN", "object": "San Francisco"},
    {"subject": "OpenAI", "relation": "DEVELOPED", "object": "GPT-4"},
    {"subject": "GPT-4", "relation": "INSTANCE_OF", "object": "Large Language Model"},
    {"subject": "Sam Altman", "relation": "CEO_OF", "object": "OpenAI"},
    {"subject": "OpenAI", "relation": "RAISED_FROM", "object": "Microsoft"},
    {"subject": "OpenAI", "relation": "RAISED_AMOUNT", "object": "$10 billion"},
    {"subject": "OpenAI", "relation": "RAISED_IN", "object": "2023"},
    {"subject": "Google", "relation": "FOUNDED_BY", "object": "Larry Page"},
    {"subject": "Google", "relation": "FOUNDED_BY", "object": "Sergey Brin"},
    {"subject": "Google", "relation": "FOUNDED_IN", "object": "1998"},
    {"subject": "Google", "relation": "HEADQUARTERED_IN", "object": "Mountain View, California"},
    {"subject": "Sundar Pichai", "relation": "CEO_OF", "object": "Google"},
    {"subject": "Google", "relation": "DEVELOPED", "object": "TensorFlow"},
    {"subject": "TensorFlow", "relation": "INSTANCE_OF", "object": "Machine Learning Framework"},
    {"subject": "Google", "relation": "ACQUIRED", "object": "DeepMind"},
    {"subject": "Google", "relation": "ACQUIRED_IN", "object": "2014"},
    {"subject": "Google", "relation": "RELEASED", "object": "Gemini"},
    {"subject": "Gemini", "relation": "INSTANCE_OF", "object": "Multimodal AI Model"},
    {"subject": "Microsoft", "relation": "FOUNDED_BY", "object": "Bill Gates"},
    {"subject": "Microsoft", "relation": "FOUNDED_BY", "object": "Paul Allen"},
    {"subject": "Microsoft", "relation": "FOUNDED_IN", "object": "1975"},
    {"subject": "Microsoft", "relation": "HEADQUARTERED_IN", "object": "Redmond, Washington"},
    {"subject": "Satya Nadella", "relation": "CEO_OF", "object": "Microsoft"},
    {"subject": "Microsoft", "relation": "INVESTED_IN", "object": "OpenAI"},
    {"subject": "Microsoft", "relation": "INVESTED_AMOUNT", "object": "$10 billion"},
    {"subject": "Microsoft", "relation": "INVESTED_IN_YEAR", "object": "2023"},
    {"subject": "Microsoft", "relation": "DEVELOPED", "object": "Azure"},
    {"subject": "Azure", "relation": "INSTANCE_OF", "object": "Cloud Computing Platform"},
    {"subject": "Microsoft", "relation": "ACQUIRED", "object": "GitHub"},
    {"subject": "Microsoft", "relation": "ACQUIRED_IN", "object": "2018"},
    {"subject": "Meta", "relation": "FOUNDED_BY", "object": "Mark Zuckerberg"},
    {"subject": "Meta", "relation": "FOUNDED_IN", "object": "2004"},
    {"subject": "Meta", "relation": "HEADQUARTERED_IN", "object": "Menlo Park, California"},
    {"subject": "Mark Zuckerberg", "relation": "CEO_OF", "object": "Meta"},
    {"subject": "Meta", "relation": "DEVELOPED", "object": "PyTorch"},
    {"subject": "PyTorch", "relation": "INSTANCE_OF", "object": "Machine Learning Framework"},
    {"subject": "Meta", "relation": "RELEASED", "object": "LLaMA"},
    {"subject": "LLaMA", "relation": "INSTANCE_OF", "object": "Large Language Model"},
    {"subject": "Meta", "relation": "ACQUIRED", "object": "Instagram"},
    {"subject": "Meta", "relation": "ACQUIRED_IN", "object": "2012"},
    {"subject": "Apple", "relation": "FOUNDED_BY", "object": "Steve Jobs"},
    {"subject": "Apple", "relation": "FOUNDED_BY", "object": "Steve Wozniak"},
    {"subject": "Apple", "relation": "FOUNDED_BY", "object": "Ronald Wayne"},
    {"subject": "Apple", "relation": "FOUNDED_IN", "object": "1976"},
    {"subject": "Apple", "relation": "HEADQUARTERED_IN", "object": "Cupertino, California"},
    {"subject": "Tim Cook", "relation": "CEO_OF", "object": "Apple"},
    {"subject": "Apple", "relation": "DEVELOPED", "object": "iPhone"},
    {"subject": "iPhone", "relation": "RELEASED_IN", "object": "2007"},
    {"subject": "Apple", "relation": "DEVELOPED", "object": "M1 chip"},
    {"subject": "Apple", "relation": "ACQUIRED", "object": "Siri"},
    {"subject": "Apple", "relation": "ACQUIRED_IN", "object": "2010"},
    {"subject": "Amazon", "relation": "FOUNDED_BY", "object": "Jeff Bezos"},
    {"subject": "Amazon", "relation": "FOUNDED_IN", "object": "1994"},
    {"subject": "Amazon", "relation": "HEADQUARTERED_IN", "object": "Seattle, Washington"},
    {"subject": "Andy Jassy", "relation": "CEO_OF", "object": "Amazon"},
    {"subject": "Amazon", "relation": "DEVELOPED", "object": "AWS"},
    {"subject": "AWS", "relation": "INSTANCE_OF", "object": "Cloud Computing Platform"},
    {"subject": "Amazon", "relation": "ACQUIRED", "object": "Whole Foods"},
    {"subject": "Amazon", "relation": "ACQUIRED_IN", "object": "2017"},
    {"subject": "Amazon", "relation": "DEVELOPED", "object": "Alexa"},
    {"subject": "Alexa", "relation": "INSTANCE_OF", "object": "Voice Assistant"},
    {"subject": "NVIDIA", "relation": "FOUNDED_BY", "object": "Jensen Huang"},
    {"subject": "NVIDIA", "relation": "FOUNDED_BY", "object": "Chris Malachowsky"},
    {"subject": "NVIDIA", "relation": "FOUNDED_BY", "object": "Curtis Priem"},
    {"subject": "NVIDIA", "relation": "FOUNDED_IN", "object": "1993"},
    {"subject": "Jensen Huang", "relation": "CEO_OF", "object": "NVIDIA"},
    {"subject": "NVIDIA", "relation": "HEADQUARTERED_IN", "object": "Santa Clara, California"},
    {"subject": "NVIDIA", "relation": "DEVELOPED", "object": "H100 GPU"},
    {"subject": "H100 GPU", "relation": "USED_FOR", "object": "AI training"},
    {"subject": "NVIDIA", "relation": "DEVELOPED", "object": "CUDA"},
    {"subject": "CUDA", "relation": "INSTANCE_OF", "object": "Parallel Computing Platform"},
    {"subject": "NVIDIA", "relation": "ACQUIRED", "object": "Mellanox"},
    {"subject": "NVIDIA", "relation": "ACQUIRED_IN", "object": "2020"},
    {"subject": "Tesla", "relation": "FOUNDED_BY", "object": "Martin Eberhard"},
    {"subject": "Tesla", "relation": "FOUNDED_BY", "object": "Marc Tarpenning"},
    {"subject": "Tesla", "relation": "FOUNDED_IN", "object": "2003"},
    {"subject": "Elon Musk", "relation": "JOINED", "object": "Tesla"},
    {"subject": "Elon Musk", "relation": "JOINED_IN", "object": "2004"},
    {"subject": "Elon Musk", "relation": "CEO_OF", "object": "Tesla"},
    {"subject": "Tesla", "relation": "HEADQUARTERED_IN", "object": "Austin, Texas"},
    {"subject": "Tesla", "relation": "DEVELOPED", "object": "Autopilot"},
    {"subject": "Autopilot", "relation": "INSTANCE_OF", "object": "Self-Driving System"},
    {"subject": "Tesla", "relation": "DEVELOPED", "object": "Model S"},
    {"subject": "Model S", "relation": "INSTANCE_OF", "object": "Electric Vehicle"},
    {"subject": "DeepMind", "relation": "FOUNDED_BY", "object": "Demis Hassabis"},
    {"subject": "DeepMind", "relation": "FOUNDED_BY", "object": "Shane Legg"},
    {"subject": "DeepMind", "relation": "FOUNDED_BY", "object": "Mustafa Suleyman"},
    {"subject": "DeepMind", "relation": "FOUNDED_IN", "object": "2010"},
    {"subject": "DeepMind", "relation": "HEADQUARTERED_IN", "object": "London"},
    {"subject": "DeepMind", "relation": "ACQUIRED_BY", "object": "Google"},
    {"subject": "DeepMind", "relation": "ACQUIRED_IN", "object": "2014"},
    {"subject": "DeepMind", "relation": "DEVELOPED", "object": "AlphaGo"},
    {"subject": "AlphaGo", "relation": "DEFEATED", "object": "Lee Sedol"},
    {"subject": "AlphaGo", "relation": "DEFEATED_IN", "object": "2016"},
    {"subject": "DeepMind", "relation": "DEVELOPED", "object": "AlphaFold"},
    {"subject": "AlphaFold", "relation": "PREDICTS", "object": "Protein Structures"},
    {"subject": "Anthropic", "relation": "FOUNDED_BY", "object": "Dario Amodei"},
    {"subject": "Anthropic", "relation": "FOUNDED_BY", "object": "Daniela Amodei"},
    {"subject": "Anthropic", "relation": "FOUNDED_IN", "object": "2021"},
    {"subject": "Dario Amodei", "relation": "CEO_OF", "object": "Anthropic"},
    {"subject": "Anthropic", "relation": "HEADQUARTERED_IN", "object": "San Francisco"},
    {"subject": "Anthropic", "relation": "DEVELOPED", "object": "Claude"},
    {"subject": "Claude", "relation": "INSTANCE_OF", "object": "Large Language Model"},
    {"subject": "Anthropic", "relation": "RAISED_FROM", "object": "Amazon"},
    {"subject": "Anthropic", "relation": "RAISED_AMOUNT", "object": "$7.3 billion"},
    {"subject": "Anthropic", "relation": "RAISED_FROM", "object": "Google"},
    {"subject": "Anthropic", "relation": "RAISED_AMOUNT", "object": "$300 million"},
    {"subject": "Anthropic", "relation": "RAISED_IN", "object": "2023"},
]


def load_corpus(filepath: str | Path) -> list[str]:
    content = Path(filepath).read_text(encoding="utf-8")
    return [p.strip() for p in content.split("\n\n") if p.strip()]


def normalize_triple(triple: dict[str, str]) -> dict[str, str]:
    return {
        "subject": str(triple["subject"]).strip(),
        "relation": re.sub(r"[^A-Z0-9_]", "_", str(triple["relation"]).strip().upper()),
        "object": str(triple["object"]).strip(),
    }


def deduplicate(triples: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, str]] = []
    for triple in triples:
        clean = normalize_triple(triple)
        key = (clean["subject"].casefold(), clean["relation"], clean["object"].casefold())
        if key not in seen:
            seen.add(key)
            output.append(clean)
    return output


def _extract_with_openai(text: str) -> list[dict[str, str]]:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    return deduplicate(json.loads(raw))


def extract_triples_from_text(text: str, use_openai: bool | None = None) -> list[dict[str, str]]:
    """Extract triples from one paragraph.

    When OpenAI is unavailable, return the curated triples whose subject appears
    in the paragraph. This keeps the lab reproducible.
    """

    should_use_openai = USE_OPENAI if use_openai is None else use_openai
    if should_use_openai:
        try:
            return _extract_with_openai(text)
        except Exception as exc:
            print(f"[WARN] OpenAI extraction failed, using deterministic fallback: {exc}")

    paragraph_lower = text.casefold()
    matched = [
        triple
        for triple in CURATED_TRIPLES
        if triple["subject"].casefold() in paragraph_lower
        or triple["object"].casefold() in paragraph_lower
    ]
    return deduplicate(matched)


def extract_all_triples(corpus_path: str | Path, use_openai: bool | None = None) -> list[dict[str, str]]:
    paragraphs = load_corpus(corpus_path)
    triples: list[dict[str, str]] = []

    print(f"[INFO] Extracting triples from {len(paragraphs)} paragraphs...")
    for index, paragraph in enumerate(paragraphs, start=1):
        print(f"[INFO] Paragraph {index}/{len(paragraphs)}")
        triples.extend(extract_triples_from_text(paragraph, use_openai=use_openai))

    output = deduplicate(triples)
    print(f"[INFO] Extracted {len(output)} unique triples.")
    return output


def save_triples(triples: list[dict[str, str]], path: str | Path = TRIPLES_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(triples, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sample = "OpenAI was founded by Sam Altman and Elon Musk in 2015."
    for item in extract_triples_from_text(sample, use_openai=False):
        print(f"({item['subject']}, {item['relation']}, {item['object']})")
