"""Graph construction backends.

The lab supports two modes:
- NetworkXGraphBuilder: default, local and reproducible.
- Neo4jGraphBuilder: optional, for Neo4j Browser/Bloom visualization.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Protocol

import networkx as nx

from config import (
    DEFAULT_GRAPH_BACKEND,
    GRAPH_IMAGE_PATH,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)
from rag.embeddings import build_node_embeddings, save_node_embeddings


class GraphBackend(Protocol):
    def clear_graph(self) -> None: ...
    def insert_triples(self, triples: list[dict[str, str]]) -> None: ...
    def get_neighbors(self, entity_name: str, max_hops: int = 2) -> list[dict[str, str]]: ...
    def get_all_entities(self) -> list[str]: ...
    def get_all_triples(self) -> list[dict[str, str]]: ...
    def get_entity_embeddings(self) -> dict[str, list[float]]: ...
    def get_stats(self) -> dict[str, int]: ...
    def close(self) -> None: ...


def validate_relation(relation: str) -> str:
    relation = relation.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", relation):
        raise ValueError(f"Invalid relation type: {relation!r}")
    return relation


class NetworkXGraphBuilder:
    """In-memory directed multigraph for offline GraphRAG."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.node_embeddings: dict[str, dict[str, object]] = {}

    def clear_graph(self) -> None:
        self.graph.clear()

    def insert_triples(self, triples: list[dict[str, str]]) -> None:
        for triple in triples:
            subject = triple["subject"].strip()
            relation = validate_relation(triple["relation"])
            obj = triple["object"].strip()
            self.graph.add_node(subject)
            self.graph.add_node(obj)
            if not self.graph.has_edge(subject, obj, key=relation):
                self.graph.add_edge(subject, obj, key=relation, relation=relation)
        self.node_embeddings = build_node_embeddings(triples)
        for node, payload in self.node_embeddings.items():
            if node in self.graph:
                self.graph.nodes[node]["embedding"] = payload["embedding"]
                self.graph.nodes[node]["embedding_text"] = payload["text"]
        save_node_embeddings(self.node_embeddings)
        print(f"[INFO] Imported {len(triples)} triples into NetworkX.")
        print(f"[INFO] Added embeddings for {len(self.node_embeddings)} nodes.")

    def get_neighbors(self, entity_name: str, max_hops: int = 2) -> list[dict[str, str]]:
        if entity_name not in self.graph:
            return []

        undirected = self.graph.to_undirected(as_view=True)
        lengths = nx.single_source_shortest_path_length(undirected, entity_name, cutoff=max_hops)
        nodes = set(lengths)

        triples: list[dict[str, str]] = []
        for subject, obj, _, data in self.graph.edges(keys=True, data=True):
            if subject in nodes and obj in nodes:
                triples.append(
                    {"subject": subject, "relation": data["relation"], "object": obj}
                )
        return sorted(triples, key=lambda item: (item["subject"], item["relation"], item["object"]))

    def get_all_entities(self) -> list[str]:
        return sorted(str(node) for node in self.graph.nodes)

    def get_all_triples(self) -> list[dict[str, str]]:
        triples = [
            {"subject": subject, "relation": data["relation"], "object": obj}
            for subject, obj, _, data in self.graph.edges(keys=True, data=True)
        ]
        return sorted(triples, key=lambda item: (item["subject"], item["relation"], item["object"]))

    def get_entity_embeddings(self) -> dict[str, list[float]]:
        return {
            node: payload["embedding"]
            for node, payload in self.node_embeddings.items()
            if isinstance(payload.get("embedding"), list)
        }

    def get_stats(self) -> dict[str, int]:
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}

    def export_image(self, path: str | Path = GRAPH_IMAGE_PATH) -> Path:
        """Save a readable Matplotlib knowledge graph screenshot for the report.

        A full 100-node spring layout is technically complete but hard to read.
        The report image uses company-centric panels so each relation remains
        legible while still showing the constructed graph.
        """

        import matplotlib.pyplot as plt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        company_nodes = [
            "OpenAI",
            "Google",
            "Microsoft",
            "Meta",
            "Apple",
            "Amazon",
            "NVIDIA",
            "Tesla",
            "DeepMind",
            "Anthropic",
        ]
        relation_priority = {
            "CEO_OF",
            "FOUNDED_BY",
            "FOUNDED_IN",
            "HEADQUARTERED_IN",
            "DEVELOPED",
            "RELEASED",
            "ACQUIRED",
            "INVESTED_IN",
            "RAISED_FROM",
        }

        fig, axes = plt.subplots(2, 5, figsize=(24, 12))
        fig.suptitle("Tech Company Knowledge Graph", fontsize=22, weight="bold", y=0.98)

        for ax, company in zip(axes.flat, company_nodes):
            direct_edges = []
            for subject, obj, _, data in self.graph.edges(keys=True, data=True):
                relation = data["relation"]
                if relation not in relation_priority:
                    continue
                if subject == company or obj == company:
                    direct_edges.append((subject, obj, relation))

            # Keep panels readable: show the most report-relevant relations first.
            direct_edges = sorted(
                direct_edges,
                key=lambda item: (
                    item[2] not in {"CEO_OF", "FOUNDED_BY", "DEVELOPED", "ACQUIRED"},
                    item[2],
                    item[0],
                    item[1],
                ),
            )[:10]

            subgraph = nx.MultiDiGraph()
            for subject, obj, relation in direct_edges:
                subgraph.add_edge(subject, obj, key=relation, relation=relation)

            neighbors = sorted(node for node in subgraph.nodes if node != company)
            angle_step = 2 * math.pi / max(len(neighbors), 1)
            pos = {company: (0.0, 0.0)}
            for index, node in enumerate(neighbors):
                angle = index * angle_step + math.pi / 12
                pos[node] = (1.8 * math.cos(angle), 1.25 * math.sin(angle))

            node_colors = ["#1976a2" if node == company else "#dbeafe" for node in subgraph.nodes]
            node_sizes = [1800 if node == company else 900 for node in subgraph.nodes]

            nx.draw_networkx_nodes(
                subgraph,
                pos,
                ax=ax,
                node_color=node_colors,
                node_size=node_sizes,
                edgecolors="#123047",
                linewidths=1.0,
            )
            nx.draw_networkx_edges(
                subgraph,
                pos,
                ax=ax,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=14,
                width=1.0,
                edge_color="#6b7280",
                connectionstyle="arc3,rad=0.08",
            )
            nx.draw_networkx_labels(
                subgraph,
                pos,
                ax=ax,
                font_size=8,
                font_family="DejaVu Sans",
                font_weight="bold",
            )

            edge_labels = {
                (subject, obj, relation): relation.replace("_", "\n")
                for subject, obj, relation in direct_edges
            }
            nx.draw_networkx_edge_labels(
                subgraph,
                pos,
                ax=ax,
                edge_labels=edge_labels,
                font_size=6,
                rotate=False,
                label_pos=0.55,
                bbox={"alpha": 0.75, "color": "white", "pad": 0.15},
            )

            ax.set_title(company, fontsize=13, weight="bold", pad=8)
            ax.set_xlim(-2.4, 2.4)
            ax.set_ylim(-1.8, 1.8)
            ax.axis("off")

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(path, dpi=180)
        plt.close()
        return path

    def close(self) -> None:
        return None


class Neo4jGraphBuilder:
    """Neo4j backend for optional browser visualization."""

    def __init__(self) -> None:
        from neo4j import GraphDatabase
        from neo4j.exceptions import AuthError, ServiceUnavailable

        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        try:
            self.driver.verify_connectivity()
        except ServiceUnavailable as exc:
            self.driver.close()
            raise RuntimeError(
                "Cannot connect to Neo4j at "
                f"{NEO4J_URI}. Open Neo4j Desktop, start your database, "
                "then verify that Bolt is enabled on port 7687."
            ) from exc
        except AuthError as exc:
            self.driver.close()
            raise RuntimeError(
                "Neo4j authentication failed. Check NEO4J_USERNAME and "
                "NEO4J_PASSWORD in your .env file."
            ) from exc
        print("[INFO] Connected to Neo4j.")

    def close(self) -> None:
        self.driver.close()

    def clear_graph(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def insert_triples(self, triples: list[dict[str, str]]) -> None:
        node_embeddings = build_node_embeddings(triples)
        save_node_embeddings(node_embeddings)

        with self.driver.session() as session:
            for triple in triples:
                relation = validate_relation(triple["relation"])
                query = f"""
                MERGE (a:Entity {{name: $subject}})
                MERGE (b:Entity {{name: $object}})
                MERGE (a)-[:{relation}]->(b)
                """
                session.run(query, subject=triple["subject"], object=triple["object"])
            for node, payload in node_embeddings.items():
                session.run(
                    """
                    MATCH (n:Entity {name: $name})
                    SET n.embedding = $embedding,
                        n.embedding_text = $embedding_text
                    """,
                    name=node,
                    embedding=payload["embedding"],
                    embedding_text=payload["text"],
                )
        print(f"[INFO] Imported {len(triples)} triples into Neo4j.")
        print(f"[INFO] Added embeddings for {len(node_embeddings)} nodes.")

    def get_neighbors(self, entity_name: str, max_hops: int = 2) -> list[dict[str, str]]:
        with self.driver.session() as session:
            query = f"""
            MATCH path = (start:Entity {{name: $name}})-[*1..{max_hops}]-(neighbor:Entity)
            UNWIND relationships(path) AS rel
            RETURN DISTINCT
                startNode(rel).name AS subject,
                type(rel) AS relation,
                endNode(rel).name AS object
            ORDER BY subject, relation, object
            """
            records = session.run(query, name=entity_name)
            return [
                {"subject": row["subject"], "relation": row["relation"], "object": row["object"]}
                for row in records
            ]

    def get_all_entities(self) -> list[str]:
        with self.driver.session() as session:
            records = session.run("MATCH (n:Entity) RETURN n.name AS name ORDER BY name")
            return [row["name"] for row in records]

    def get_all_triples(self) -> list[dict[str, str]]:
        with self.driver.session() as session:
            records = session.run(
                """
                MATCH (a:Entity)-[r]->(b:Entity)
                RETURN a.name AS subject, type(r) AS relation, b.name AS object
                ORDER BY subject, relation, object
                """
            )
            return [
                {"subject": row["subject"], "relation": row["relation"], "object": row["object"]}
                for row in records
            ]

    def get_entity_embeddings(self) -> dict[str, list[float]]:
        with self.driver.session() as session:
            records = session.run(
                """
                MATCH (n:Entity)
                WHERE n.embedding IS NOT NULL
                RETURN n.name AS name, n.embedding AS embedding
                """
            )
            return {row["name"]: row["embedding"] for row in records}

    def get_stats(self) -> dict[str, int]:
        with self.driver.session() as session:
            nodes = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()["count"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        return {"nodes": nodes, "edges": edges}


def build_graph(
    triples: list[dict[str, str]],
    backend: str = DEFAULT_GRAPH_BACKEND,
    clear: bool = True,
) -> GraphBackend:
    if backend == "neo4j":
        builder: GraphBackend = Neo4jGraphBuilder()
    elif backend in {"networkx", "local"}:
        builder = NetworkXGraphBuilder()
    else:
        raise ValueError(f"Unsupported graph backend: {backend}")

    if clear:
        builder.clear_graph()
    builder.insert_triples(triples)
    return builder


if __name__ == "__main__":
    from extraction.entity_extractor import CURATED_TRIPLES

    graph = build_graph(CURATED_TRIPLES, backend="networkx")
    print(graph.get_stats())
    if isinstance(graph, NetworkXGraphBuilder):
        print(graph.export_image())
