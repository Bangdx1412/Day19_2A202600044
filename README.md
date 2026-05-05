# Lab 19: GraphRAG với Tech Company Corpus

Dự án này hoàn thiện các yêu cầu trong slide thực hành: entity extraction, build graph, GraphRAG retrieval, benchmark GraphRAG vs Flat RAG, visualization và report.

## Checklist yêu cầu

| Yêu cầu | Trạng thái | File/artefact |
| --- | --- | --- |
| Entity extraction: output triples `(subject, predicate, object)` | Done | `extraction/entity_extractor.py`, `data/triples.json` |
| Build graph bằng NetworkX hoặc Neo4j | Done | `graph/build_neo4j.py` |
| Thêm embeddings cho nodes | Done | `data/node_embeddings.json`, Neo4j property `Entity.embedding` |
| GraphRAG retrieval: query -> seed nodes -> BFS -> subgraph-to-text -> answer | Done | `query/graph_query.py` |
| Flat RAG baseline bằng ChromaDB | Done | `rag/flat_rag.py` |
| Benchmark GraphRAG vs Flat RAG trên 20 câu hỏi | Done | `evaluation/benchmark.py`, `evaluation/benchmark_results.csv` |
| Đo accuracy, latency, cost | Done | `evaluation/benchmark_summary.md` |
| Phân tích failure modes | Done | `evaluation/benchmark_summary.md` |
| Visualization NetworkX/Neo4j | Done | `outputs/knowledge_graph.png`, `outputs/neo4j_report_queries.cypher` |

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Các thư viện chính gồm `networkx`, `neo4j`, `chromadb`, `pandas`, `matplotlib`.

## Chạy toàn bộ pipeline offline

```bash
python main.py --step all --backend networkx
```

Pipeline offline dùng extractor và embedding deterministic để dễ chấm/nộp mà không cần API.
Flat RAG vẫn dùng ChromaDB, nhưng embedding được tạo local deterministic nên không cần OpenAI key.

## Chạy từng bước

### 1. Entity extraction

```bash
python main.py --step 1
```

Output:

- `data/triples.json`

### 2. Build graph và node embeddings

NetworkX:

```bash
python main.py --step 2 --backend networkx
```

Neo4j Desktop:

```bash
python main.py --step 2 --backend neo4j
```

Output:

- `data/node_embeddings.json`
- `outputs/knowledge_graph.png` nếu dùng NetworkX
- Neo4j nodes có property `embedding` và `embedding_text` nếu dùng Neo4j

### 3. GraphRAG query

```bash
python main.py --step 3
```

Ví dụ:

```text
What company did Google acquire that developed AlphaGo?
What is the connection between NVIDIA and AI model training?
```

### 4. Benchmark

```bash
python main.py --step 4
```

Hoặc chạy trực tiếp file benchmark:

```bash
python evaluation/benchmark.py
```

Output:

- `evaluation/benchmark_results.csv`
- `evaluation/benchmark_summary.md`

Report gồm accuracy, latency, estimated cost và failure modes. CSV cũng có cột `flatrag_retriever`, giá trị bình thường là `chromadb`.

## Dùng Neo4j Desktop

1. Mở Neo4j Desktop.
2. Start local database.
3. Kiểm tra URL`.
4. Cập nhật `.env`:

```env
GRAPH_BACKEND=neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

5. Import graph:

```bash
python main.py --step 2 --backend neo4j
```

Nếu gặp lỗi `Unable to retrieve routing information`, hãy dùng `bolt://localhost:7687` thay vì `neo4j://127.0.0.1:7687` trong `.env`.

6. Mở Neo4j Browser và chạy các query trong:

```text
outputs/neo4j_report_queries.cypher
```

Kiểm tra node embeddings:

```cypher
MATCH (n:Entity)
WHERE n.embedding IS NOT NULL
RETURN n.name, size(n.embedding) AS embedding_dim
LIMIT 20;
```

## Cấu trúc project

```text
lab19_graphrag/
  config.py
  main.py
  data/
    tech_corpus.txt
    triples.json
    node_embeddings.json
  extraction/
    entity_extractor.py
  graph/
    build_neo4j.py
  query/
    graph_query.py
  rag/
    embeddings.py
    flat_rag.py
  evaluation/
    benchmark.py
    benchmark_results.csv
    benchmark_summary.md
  outputs/
    knowledge_graph.png
    neo4j_report_queries.cypher
```
