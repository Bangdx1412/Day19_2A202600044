"""CLI entry point for Lab Day 19 GraphRAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import CORPUS_PATH, DEFAULT_GRAPH_BACKEND, GRAPH_IMAGE_PATH, TRIPLES_PATH
from evaluation.benchmark import run_benchmark
from extraction.entity_extractor import extract_all_triples, save_triples
from graph.build_neo4j import NetworkXGraphBuilder, build_graph
from query.graph_query import answer_with_graph


def load_triples(path: Path = TRIPLES_PATH) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 5:
        triples = extract_all_triples(CORPUS_PATH, use_openai=False)
        save_triples(triples, path)
        return triples
    return json.loads(path.read_text(encoding="utf-8"))


def step1_extract(use_openai: bool = False) -> list[dict[str, str]]:
    print("\nSTEP 1 - Entity and relation extraction")
    triples = extract_all_triples(CORPUS_PATH, use_openai=use_openai)
    save_triples(triples)
    print(f"[DONE] Saved {len(triples)} triples to {TRIPLES_PATH}")
    for triple in triples[:8]:
        print(f"  ({triple['subject']}, {triple['relation']}, {triple['object']})")
    return triples


def step2_build_graph(backend: str = "networkx", export_image: bool = True):
    print("\nSTEP 2 - Graph construction")
    triples = load_triples()
    graph = build_graph(triples, backend=backend)
    stats = graph.get_stats()
    print(f"[DONE] Nodes: {stats['nodes']} | Edges: {stats['edges']}")

    if export_image and isinstance(graph, NetworkXGraphBuilder):
        image_path = graph.export_image(GRAPH_IMAGE_PATH)
        print(f"[DONE] Graph image saved to {image_path}")

    return graph


def step3_interactive_query(backend: str = "networkx") -> None:
    print("\nSTEP 3 - Interactive GraphRAG query")
    graph = step2_build_graph(backend=backend, export_image=False)
    print("Type a question in English, or type 'quit' to exit.")
    try:
        while True:
            question = input("\nQuestion: ").strip()
            if question.casefold() in {"q", "quit", "exit"}:
                break
            if not question:
                continue
            result = answer_with_graph(question, graph)
            print(f"Entity: {result['entity_found']}")
            print(f"Retrieved triples: {result['hop_count']}")
            print(f"Answer: {result['answer']}")
    finally:
        graph.close()


def step4_benchmark() -> None:
    print("\nSTEP 4 - Evaluation")
    run_benchmark()


def run_all(backend: str = "networkx", use_openai: bool = False) -> None:
    step1_extract(use_openai=use_openai)
    graph = step2_build_graph(backend=backend, export_image=True)
    graph.close()
    step4_benchmark()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab Day 19 GraphRAG pipeline")
    parser.add_argument(
        "--step",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--backend",
        choices=["networkx", "neo4j"],
        default=DEFAULT_GRAPH_BACKEND,
        help="Graph backend. Use neo4j only when a database is running.",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI for extraction/answering when OPENAI_API_KEY is set.",
    )
    args = parser.parse_args()

    if args.step == "1":
        step1_extract(use_openai=args.use_openai)
    elif args.step == "2":
        graph = step2_build_graph(backend=args.backend)
        graph.close()
    elif args.step == "3":
        step3_interactive_query(backend=args.backend)
    elif args.step == "4":
        step4_benchmark()
    else:
        run_all(backend=args.backend, use_openai=args.use_openai)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
