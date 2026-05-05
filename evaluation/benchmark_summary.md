# Báo cáo Lab 19 - GraphRAG với Tech Company Corpus

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
| GraphRAG | 100.0% | 0.0003s | 3034 | 206 | $0.000578 |
| Flat RAG | 60.0% | 0.0006s | 3832 | 232 | $0.000715 |

Accuracy được tính bằng cách so khớp keyword quan trọng với ground truth. Latency được đo bằng `time.perf_counter()`. Chi phí là ước tính dựa trên số input/output tokens, dùng giá cấu hình `INPUT_COST_PER_1K_TOKENS=0.00015` và `OUTPUT_COST_PER_1K_TOKENS=0.0006`. Vì lần chạy này dùng pipeline offline nên không phát sinh chi phí API thật.

## 4. Trường hợp Flat RAG sai

| Câu | GraphRAG | Flat RAG | Phân tích lỗi |
| ---: | --- | --- | --- |
| 6 | True | False | Flat RAG truy xuất được các đoạn có vẻ liên quan nhưng bỏ lỡ phép nối multi-hop.<br>What company did Google acquire that developed AlphaGo? |
| 8 | True | False | Flat RAG truy xuất được các đoạn có vẻ liên quan nhưng bỏ lỡ phép nối multi-hop.<br>What framework did the company behind Instagram develop? |
| 10 | True | False | Flat RAG truy xuất được các đoạn có vẻ liên quan nhưng bỏ lỡ phép nối multi-hop.<br>What did the company that Elon Musk co-founded develop? |
| 11 | True | False | Flat RAG truy xuất được các đoạn có vẻ liên quan nhưng bỏ lỡ phép nối multi-hop.<br>What are the AI models developed by companies headquartered in San Francisco? |
| 13 | True | False | Flat RAG truy xuất được các đoạn có vẻ liên quan nhưng bỏ lỡ phép nối multi-hop.<br>What did Google acquire and what did those acquired companies develop? |
| 14 | True | False | Flat RAG truy xuất được các đoạn có vẻ liên quan nhưng bỏ lỡ phép nối multi-hop.<br>Which cloud platforms were developed by major tech companies? |
| 16 | True | False | Flat RAG truy xuất được các đoạn có vẻ liên quan nhưng bỏ lỡ phép nối multi-hop.<br>What company acquired a firm that defeated a world champion in Go? |
| 19 | True | False | Flat RAG truy xuất được các đoạn có vẻ liên quan nhưng bỏ lỡ phép nối multi-hop.<br>Which CEO leads a company that has invested in an AI safety organization? |

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
