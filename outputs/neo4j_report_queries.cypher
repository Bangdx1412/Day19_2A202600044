// Lab Day 19 - Neo4j/Bloom report-friendly views
// Run one query at a time in Neo4j Browser, then screenshot the graph result.

// 1. Overview: company -> product/acquisition/investment only
MATCH p=(company:Entity)-[r:DEVELOPED|RELEASED|ACQUIRED|INVESTED_IN|RAISED_FROM]->(target:Entity)
WHERE company.name IN [
  "OpenAI", "Google", "Microsoft", "Meta", "Apple",
  "Amazon", "NVIDIA", "Tesla", "DeepMind", "Anthropic"
]
RETURN p
LIMIT 80;

// 2. OpenAI 2-hop neighborhood
MATCH p=(n:Entity {name: "OpenAI"})-[*1..2]-(m:Entity)
RETURN p
LIMIT 50;

// 3. Google and DeepMind multi-hop story
MATCH p=(n:Entity {name: "Google"})-[*1..3]-(m:Entity)
WHERE m.name IN ["DeepMind", "AlphaGo", "AlphaFold", "Lee Sedol", "Gemini", "TensorFlow"]
RETURN p
LIMIT 50;

// 4. Founders and CEOs only
MATCH p=(person:Entity)-[r:CEO_OF]->(company:Entity)
RETURN p
UNION
MATCH p=(company:Entity)-[r:FOUNDED_BY]->(person:Entity)
RETURN p
LIMIT 80;

// 5. Cloud and AI infrastructure
MATCH p=(company:Entity)-[r:DEVELOPED]->(tech:Entity)-[*0..1]-(kind:Entity)
WHERE tech.name IN ["AWS", "Azure", "H100 GPU", "CUDA", "TensorFlow", "PyTorch"]
RETURN p
LIMIT 60;

// 6. Node embeddings check
MATCH (n:Entity)
WHERE n.embedding IS NOT NULL
RETURN n.name AS node, size(n.embedding) AS embedding_dim
ORDER BY node
LIMIT 20;

// 7. Count check for report
MATCH (n:Entity)
WITH count(n) AS nodes
MATCH ()-[r]->()
RETURN nodes, count(r) AS edges;
