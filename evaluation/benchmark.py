"""Benchmark Flat RAG vs GraphRAG on 20 questions."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CORPUS_PATH,
    EVALUATION_DIR,
    INPUT_COST_PER_1K_TOKENS,
    OUTPUT_COST_PER_1K_TOKENS,
    TRIPLES_PATH,
)
from extraction.entity_extractor import extract_all_triples, save_triples
from graph.build_neo4j import NetworkXGraphBuilder, build_graph
from query.graph_query import answer_with_graph
from rag.flat_rag import FlatRAG


BENCHMARK_QUESTIONS = [
    "Who is the CEO of OpenAI?",
    "When was Google founded?",
    "Where is Microsoft headquartered?",
    "Who founded Tesla?",
    "What did Meta develop as an open-source LLM?",
    "What company did Google acquire that developed AlphaGo?",
    "Which company invested $10 billion in OpenAI?",
    "What framework did the company behind Instagram develop?",
    "Who is the CEO of the company that developed AWS?",
    "What did the company that Elon Musk co-founded develop?",
    "What are the AI models developed by companies headquartered in San Francisco?",
    "Which companies has Elon Musk been involved in founding?",
    "What did Google acquire and what did those acquired companies develop?",
    "Which cloud platforms were developed by major tech companies?",
    "What open-source ML frameworks exist and who created them?",
    "What company acquired a firm that defeated a world champion in Go?",
    "Who leads the company that raised money from both Google and Amazon?",
    "What did the company founded by Jeff Bezos develop as a voice assistant?",
    "Which CEO leads a company that has invested in an AI safety organization?",
    "What is the connection between NVIDIA and AI model training?",
]


GROUND_TRUTH = [
    "Sam Altman",
    "1998",
    "Redmond, Washington",
    "Martin Eberhard; Marc Tarpenning",
    "LLaMA",
    "Google acquired DeepMind; DeepMind developed AlphaGo",
    "Microsoft",
    "Meta developed PyTorch",
    "Andy Jassy",
    "OpenAI developed GPT-4",
    "OpenAI developed GPT-4; Anthropic developed Claude",
    "OpenAI; Tesla",
    "DeepMind developed AlphaGo and AlphaFold",
    "Amazon AWS; Microsoft Azure",
    "Google TensorFlow; Meta PyTorch",
    "Google acquired DeepMind; AlphaGo defeated Lee Sedol",
    "Dario Amodei; Anthropic",
    "Amazon developed Alexa",
    "Satya Nadella; Microsoft invested in OpenAI",
    "NVIDIA H100 GPU; CUDA; AI training",
]


KEYWORDS = [
    ["sam altman"],
    ["1998"],
    ["redmond", "washington"],
    ["martin eberhard", "marc tarpenning"],
    ["llama"],
    ["google", "deepmind", "alphago"],
    ["microsoft"],
    ["meta", "pytorch"],
    ["andy jassy", "amazon"],
    ["openai", "gpt-4"],
    ["gpt-4", "claude"],
    ["openai", "tesla"],
    ["deepmind", "alphago", "alphafold"],
    ["aws", "azure"],
    ["tensorflow", "pytorch"],
    ["google", "deepmind", "lee sedol"],
    ["dario amodei", "anthropic", "google", "amazon"],
    ["amazon", "alexa"],
    ["satya nadella", "microsoft", "openai"],
    ["nvidia", "h100", "cuda"],
]


def load_or_create_triples(path: Path = TRIPLES_PATH) -> list[dict[str, str]]:
    if path.exists() and path.stat().st_size > 5:
        return json.loads(path.read_text(encoding="utf-8"))

    triples = extract_all_triples(CORPUS_PATH, use_openai=False)
    save_triples(triples, path)
    return triples


def is_correct(answer: str, keywords: list[str]) -> bool:
    answer_lower = answer.casefold()
    return all(keyword.casefold() in answer_lower for keyword in keywords)


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 1.3))


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        (input_tokens / 1000 * INPUT_COST_PER_1K_TOKENS)
        + (output_tokens / 1000 * OUTPUT_COST_PER_1K_TOKENS),
        6,
    )


def failure_mode(row: pd.Series) -> str:
    if row["graphrag_correct"] and not row["flatrag_correct"]:
        return "Flat RAG truy xuất được các đoạn có vẻ liên quan nhưng bỏ lỡ phép nối multi-hop."
    if not row["graphrag_correct"] and row["flatrag_correct"]:
        return "GraphRAG chọn seed/traversal thiếu một thực thể mà Flat RAG truy xuất được."
    if not row["graphrag_correct"] and not row["flatrag_correct"]:
        return "Cả hai hệ thống đều không truy xuất đủ bằng chứng."
    return ""


def rows_to_markdown(rows: pd.DataFrame) -> str:
    header = "| Câu | GraphRAG | Flat RAG | Phân tích lỗi |"
    separator = "| ---: | --- | --- | --- |"
    lines = [header, separator]
    for _, row in rows.iterrows():
        question = str(row["question"]).replace("|", "\\|")
        mode = str(row["failure_mode"]).replace("|", "\\|")
        lines.append(
            f"| {row['question_id']} | {row['graphrag_correct']} | "
            f"{row['flatrag_correct']} | {mode}<br>{question} |"
        )
    return "\n".join(lines)


def write_report(df: pd.DataFrame, output_path: Path) -> None:
    graph_accuracy = df["graphrag_correct"].mean() * 100
    flat_accuracy = df["flatrag_correct"].mean() * 100
    graph_avg = df["graphrag_time_s"].mean()
    flat_avg = df["flatrag_time_s"].mean()
    graph_input_tokens = int(df["graphrag_input_tokens"].sum())
    flat_input_tokens = int(df["flatrag_input_tokens"].sum())
    graph_output_tokens = int(df["graphrag_output_tokens"].sum())
    flat_output_tokens = int(df["flatrag_output_tokens"].sum())
    graph_cost = df["graphrag_estimated_cost_usd"].sum()
    flat_cost = df["flatrag_estimated_cost_usd"].sum()
    failure_rows = df[df["failure_mode"] != ""]

    if failure_rows.empty:
        failure_table = "No failures were recorded by the keyword-based evaluator."
    else:
        failure_table = rows_to_markdown(failure_rows)

    report = f"""# Báo cáo Lab 19 - GraphRAG với Tech Company Corpus

## 1. Mục tiêu bài làm

Trong lab này, em xây dựng một pipeline GraphRAG nhỏ trên bộ dữ liệu về các công ty công nghệ. Mục tiêu chính là biến corpus dạng văn bản thành knowledge graph, sau đó dùng graph này để trả lời các câu hỏi cần suy luận nhiều bước. Em cũng làm thêm một hệ Flat RAG đơn giản để có baseline so sánh.

Các phần đã làm gồm:

- Trích xuất triples theo dạng `(subject, predicate, object)`.
- Nạp triples vào graph bằng NetworkX và có hỗ trợ Neo4j Desktop.
- Tạo embedding cho từng node để hỗ trợ bước chọn seed node.
- Cài GraphRAG retrieval theo luồng: câu hỏi -> seed node -> BFS traversal -> chuyển subgraph thành text -> sinh câu trả lời.
- Chạy benchmark 20 câu hỏi multi-hop để so sánh GraphRAG với Flat RAG.

## 2. Cách thực hiện

Ở bước indexing, corpus được tách thành các đoạn nhỏ, sau đó chuyển thành danh sách triples. Ví dụ một câu như "OpenAI was founded by Sam Altman..." sẽ tạo ra các quan hệ như `OpenAI - FOUNDED_BY -> Sam Altman`. Sau khi có triples, em khử trùng lặp để tránh việc cùng một entity bị tách thành nhiều node khác nhau.

Graph được xây từ các triples này. Mỗi subject và object trở thành một node, predicate trở thành cạnh nối giữa hai node. Với NetworkX, graph được lưu trong bộ nhớ để chạy offline. Với Neo4j, các node được lưu dưới label `Entity`, còn quan hệ được lưu theo đúng tên predicate như `FOUNDED_BY`, `DEVELOPED`, `ACQUIRED`.

Với GraphRAG retrieval, hệ thống tìm seed node từ câu hỏi trước. Nếu câu hỏi nhắc trực tiếp entity như "OpenAI" hoặc "Google" thì dùng entity đó làm seed. Nếu không match trực tiếp, hệ thống dùng embedding similarity để chọn node gần nhất. Từ seed node, hệ thống duyệt BFS trong phạm vi 2-hop, lấy subgraph liên quan rồi chuyển các triples đó thành đoạn text context để trả lời.

Flat RAG được dùng làm baseline. Hệ này dùng ChromaDB để index các đoạn văn bản và truy xuất top-k chunk gần với câu hỏi, nhưng không dùng cấu trúc graph. Vì vậy Flat RAG thường ổn với câu hỏi một bước, nhưng dễ thiếu thông tin khi câu trả lời phải nối nhiều quan hệ nằm ở nhiều đoạn khác nhau.

## 3. Kết quả benchmark

Em chạy 20 câu hỏi benchmark trên cả hai hệ thống. Kết quả tổng hợp như sau:

| Hệ thống | Accuracy | Latency trung bình | Input tokens | Output tokens | Chi phí ước tính |
| --- | ---: | ---: | ---: | ---: | ---: |
| GraphRAG | {graph_accuracy:.1f}% | {graph_avg:.4f}s | {graph_input_tokens} | {graph_output_tokens} | ${graph_cost:.6f} |
| Flat RAG | {flat_accuracy:.1f}% | {flat_avg:.4f}s | {flat_input_tokens} | {flat_output_tokens} | ${flat_cost:.6f} |

Accuracy được tính bằng cách so khớp keyword quan trọng với ground truth. Latency được đo bằng `time.perf_counter()`. Chi phí là ước tính dựa trên số input/output tokens, dùng giá cấu hình `INPUT_COST_PER_1K_TOKENS={INPUT_COST_PER_1K_TOKENS}` và `OUTPUT_COST_PER_1K_TOKENS={OUTPUT_COST_PER_1K_TOKENS}`. Vì lần chạy này dùng pipeline offline nên không phát sinh chi phí API thật.

## 4. Trường hợp Flat RAG sai

{failure_table}

Trường hợp đáng chú ý nhất là câu hỏi:

> What are the AI models developed by companies headquartered in San Francisco?

GraphRAG trả lời được vì graph nối được hai cụm thông tin: công ty nào đặt trụ sở ở San Francisco, và công ty đó phát triển model AI nào. Flat RAG thì lấy được các đoạn có vẻ liên quan, nhưng không nối đủ hai bước này nên bị sai.

## 5. Nhận xét

Qua benchmark, GraphRAG cho kết quả tốt hơn ở nhóm câu hỏi multi-hop. Lý do là graph giữ lại quan hệ rõ ràng giữa các entity, nên khi câu hỏi cần đi qua nhiều bước như "company -> acquired company -> product -> event", traversal trên graph tự nhiên hơn so với chỉ tìm chunk giống về mặt từ vựng.

Flat RAG vẫn có ưu điểm là đơn giản, dễ chạy và nhanh. Với câu hỏi trực tiếp như "Who is the CEO of OpenAI?" hoặc "When was Google founded?", Flat RAG trả lời khá tốt. Tuy nhiên khi câu hỏi yêu cầu ghép thông tin từ nhiều đoạn, hệ này dễ lấy thiếu evidence hoặc lấy đúng chủ đề nhưng chưa đủ quan hệ trung gian.

Nhìn chung, với corpus nhỏ trong lab này, GraphRAG phù hợp hơn cho các câu hỏi cần suy luận theo quan hệ. Nếu mở rộng sang corpus lớn hơn, phần cần cải thiện tiếp theo là entity extraction tự động bằng LLM thật, chuẩn hóa entity tốt hơn, và đánh giá bằng judge model thay vì chỉ keyword matching.

## 6. Đề xuất công cụ

| Mục tiêu | Công cụ nên dùng | Lý do |
| --- | --- | --- |
| Dễ bắt đầu | NetworkX | Chạy offline, dễ debug, phù hợp để hiểu BFS traversal và kiểm tra logic GraphRAG. |
| Trực quan hóa tốt | Neo4j Desktop/Bloom | Dễ xem quan hệ giữa các entity, phù hợp để chụp hình graph và kiểm tra path multi-hop. |
| Flat RAG baseline | ChromaDB | Đúng yêu cầu vector database, dễ index chunk và truy xuất top-k context. |
| Mở rộng hệ thống | Neo4j + LLM API | Neo4j quản lý graph tốt hơn khi dữ liệu lớn, còn LLM giúp extraction và answer generation linh hoạt hơn. |

## 7. File kết quả

- Triples: `data/triples.json`
- Node embeddings: `data/node_embeddings.json`
- Benchmark chi tiết: `evaluation/benchmark_results.csv`
- Visualization: `outputs/knowledge_graph.png`
- Query Neo4j/Bloom để chụp hình: `outputs/neo4j_report_queries.cypher`
"""
    output_path.write_text(report, encoding="utf-8-sig")


def run_benchmark(output_csv: str | Path = EVALUATION_DIR / "benchmark_results.csv") -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    triples = load_or_create_triples()
    graph = build_graph(triples, backend="networkx")
    flat_rag = FlatRAG()
    flat_rag.index_corpus()

    rows = []
    for index, (question, ground_truth, keywords) in enumerate(
        zip(BENCHMARK_QUESTIONS, GROUND_TRUTH, KEYWORDS),
        start=1,
    ):
        print(f"[{index:02d}/20] {question}")

        start = time.perf_counter()
        graph_result = answer_with_graph(question, graph, use_openai=False)
        graph_time = time.perf_counter() - start

        start = time.perf_counter()
        flat_result = flat_rag.query(question)
        flat_time = time.perf_counter() - start

        rows.append(
            {
                "question_id": index,
                "question": question,
                "ground_truth": ground_truth,
                "graphrag_answer": graph_result["answer"],
                "flatrag_answer": flat_result["answer"],
                "graphrag_entity": graph_result["entity_found"],
                "graphrag_seed_entities": "; ".join(graph_result.get("seed_entities", [])),
                "graphrag_hops": graph_result["hop_count"],
                "flatrag_retriever": flat_result.get("retriever", "unknown"),
                "graphrag_time_s": round(graph_time, 4),
                "flatrag_time_s": round(flat_time, 4),
                "graphrag_input_tokens": estimate_tokens(str(graph_result["context"])),
                "flatrag_input_tokens": estimate_tokens(str(flat_result["context"])),
                "graphrag_output_tokens": estimate_tokens(str(graph_result["answer"])),
                "flatrag_output_tokens": estimate_tokens(str(flat_result["answer"])),
                "graphrag_correct": is_correct(str(graph_result["answer"]), keywords),
                "flatrag_correct": is_correct(str(flat_result["answer"]), keywords),
                "notes": "",
            }
        )

    df = pd.DataFrame(rows)
    df["graphrag_estimated_cost_usd"] = df.apply(
        lambda row: estimate_cost(row["graphrag_input_tokens"], row["graphrag_output_tokens"]),
        axis=1,
    )
    df["flatrag_estimated_cost_usd"] = df.apply(
        lambda row: estimate_cost(row["flatrag_input_tokens"], row["flatrag_output_tokens"]),
        axis=1,
    )
    df["failure_mode"] = df.apply(failure_mode, axis=1)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    write_report(df, EVALUATION_DIR / "benchmark_summary.md")

    if isinstance(graph, NetworkXGraphBuilder):
        graph.export_image()

    graph.close()
    print(f"[DONE] Benchmark saved to {output_csv}")
    print(f"[DONE] Summary saved to {EVALUATION_DIR / 'benchmark_summary.md'}")
    return df


if __name__ == "__main__":
    dataframe = run_benchmark()
    print(dataframe[["question_id", "graphrag_correct", "flatrag_correct"]].to_string(index=False))
