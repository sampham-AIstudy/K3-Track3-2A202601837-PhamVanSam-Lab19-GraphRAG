# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Phạm Văn Sâm  
**Mã học viên:** 2A202601837  
**Khóa học:** AICB-K34 · Track 3: GraphRAG  
**Ngày thực hiện:** 19/08/2026  

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*
- **Ví dụ từ dữ liệu:** Trong chunk `693b7e08c569ec6467f9::c0000` (bài báo *"Microsoft Invests $10 Billion in OpenAI to Accelerate AI Breakthroughs"*):
  > *"Microsoft announced a multi-billion dollar investment in OpenAI on January 23, 2023... Microsoft CEO Satya Nadella stated that this partnership will allow developers and organizations across industries to access the best AI infrastructure. OpenAI, founded by Sam Altman, Greg Brockman, and Ilya Sutskever, developed ChatGPT and GPT-4. The company will deploy these models across its consumer and enterprise products."*
- **Hiện tượng:** Cụm từ *"The company"* ở câu cuối cùng xuất hiện sau khi vừa nhắc đến OpenAI ở câu liền kề trước đó. Một mô hình Coreference Resolution thông thường dễ nhầm lẫn đại từ *"The company"* là OpenAI thay vì chủ ngữ chính Microsoft của bài viết.
- **Hậu quả đối với Graph:** Nếu gán sai sang OpenAI, Knowledge Graph sẽ sinh ra **False Edge** dạng `(OpenAI)-[:DEVELOPED]->(Windows/Office)` hoặc `(OpenAI)-[:USES]->(Consumer and Enterprise Products)`, làm biến dạng quyền sở hữu sản phẩm và dẫn đến suy luận sai lệch khi truy vấn.
- **Giải pháp bảo vệ (Conservative Rule):** Áp dụng nguyên tắc chỉ giải quyết khi tiền ngữ (antecedent) hoàn toàn đơn nghĩa và rõ ràng trong cùng chunk; nếu có sự mơ hồ (ambiguity), giữ nguyên văn bản gốc và ghi nhận vào `unresolved_mentions` để tránh tạo ra False Edge.

---

### 2. Entity Resolution Threshold & Lexical Guard
> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*
- **Ngưỡng cosine similarity:** `threshold = 0.90` (trên không gian embedding `sentence-transformers/all-MiniLM-L6-v2` với chuẩn hoá vector FAISS FlatIP).
- **Cặp thực thể bị Guard chặn:** 
  1. `Sam Altman` vs `Steve Altman` (Cosine Similarity = `0.884`): Bị chặn bởi **Person First-Name Guard** do trùng họ *"Altman"* nhưng khác biệt hoàn toàn về tên gọi đầu tiên.
  2. `GPT-3` vs `GPT-4` (Cosine Similarity = `0.921`): Bị chặn bởi **Digit Conflict Guard** do khác biệt về chữ số phiên bản (`3` vs `4`).
  3. `Claude` vs `Claude 2` (Cosine Similarity = `0.912`): Bị chặn do thiếu/thừa số phiên bản (`2`).
  4. `Apple` vs `Apple Music` (Cosine Similarity = `0.897`): Bị chặn bởi **Product/Service Sub-branch Guard** (tách biệt công ty mẹ và dịch vụ con).
- **Lý do chặn:** Mặc dù embedding ngữ nghĩa của các cặp này nằm rất gần nhau trong không gian vector do cùng xuất hiện trong ngữ cảnh AI/Tech, về mặt bản thể học (ontology) chúng là các thực thể độc lập. Nếu không có Lexical Guard, đồ thị sẽ hợp nhất sai phiên bản mô hình hoặc nhân sự, gây phá hủy tính chính xác của Knowledge Graph.

---

### 3. Đồ thị & Super-node Mitigation
> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*
- **Top 3 Super-nodes trong Đồ thị:**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|------|--------------|---------------------|----------------------|
| 1 | **Google** | Company | 20 |
| 2 | **Microsoft** | Company | 18 |
| 3 | **OpenAI** | Company | 15 |

- **Ưu điểm & Rủi ro của Temporal Mitigation (Cắt tỉa theo thời gian):**
  - *Ưu điểm:* 
    - Ngăn chặn triệt để hiện tượng **Graph Context Explosion** khi duyệt BFS qua các node trung tâm (Hubs) như Google hay Microsoft.
    - Giảm chi phí token gửi qua LLM (giới hạn context $\le 3,500$ ký tự) và giảm latency truy vấn.
    - Đảm bảo câu trả lời luôn cập nhật các sự kiện, hợp tác và trạng thái lãnh đạo mới nhất (recency validity).
  - *Rủi ro:* 
    - Khi người dùng hỏi về **nguồn gốc lịch sử** hoặc **chuỗi sự kiện trong quá khứ xa** (ví dụ: *"Google mua lại DeepMind vào năm nào và ban lãnh đạo sáng lập ban đầu gồm những ai?"*), việc chỉ lấy 50 cạnh mới nhất có thể cắt tỉa (prune) mất các cạnh lịch sử cũ (2014), khiến hệ thống không tìm thấy bằng chứng quá khứ.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge):

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Độ chênh lệch ($\Delta$) | Nhận xét phân tích |
|-------------------|----------|----------|--------------------------|-------------------|
| **Comprehensiveness (Multi-hop)** | 3.667 | **4.000** | **+0.333** | GraphRAG kết nối tốt hơn các quan hệ nhiều bước. |
| **Faithfulness (Multi-hop)** | 3.667 | **4.000** | **+0.333** | GraphRAG trích dẫn chính xác nguồn gốc và quan hệ thực thể. |
| **Multi-hop Reasoning (Multi-hop)**| 3.667 | **4.000** | **+0.333** | Graph Traversal BFS lần theo đúng chuỗi `FOUNDED` → `INVESTED_IN`. |
| **Comprehensiveness (Factoid)** | **5.000** | 1.000 | -4.000 | Flat RAG tìm trúng ngay chunk chứa câu trả lời trực tiếp. |
| **Latency trung bình (s)** | 15.911s | **6.549s** | **-9.362s** | GraphRAG tập trung context ngắn gọn, giảm thời gian sinh văn bản. |
| **Token usage trung bình** | **827.0** | 1419.4 | +592.4 | Flat RAG tiêu tốn ít token hơn do không mang theo cấu trúc đồ thị. |

#### Phân tích 2 Ca lỗi Điển hình:
1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công):**
   - *Question ID & Câu hỏi:* `G02` — *"Which startups were founded by former Google researchers and later received investment from Nvidia?"*
   - *Tại sao Flat RAG thất bại / gặp khó khăn?* Thông tin về việc sáng lập Cohere bởi cựu chuyên gia Google Brain và sự kiện Nvidia đầu tư vào Cohere trong vòng Series C năm 2023 nằm phân tán ở các phần khác nhau trong bài viết. Vector Search thuần túy dựa vào semantic similarity có thể chỉ lấy được một vế (hoặc chỉ thông tin đầu tư, hoặc chỉ thông tin sáng lập).
   - *GraphRAG đã giải quyết như thế nào?* GraphRAG trích xuất Seed `Google` và `Nvidia`, sau đó duyệt đồ thị 2 bước: `(Aidan Gomez)-[:WORKED_AT]->(Google)` $ightarrow$ `(Aidan Gomez)-[:FOUNDED]->(Cohere)` $\leftarrow$ `(Nvidia)-[:INVESTED_IN]->(Cohere)`. Đồ thị cung cấp đầy đủ liên kết và câu trả lời đạt điểm tối đa (5/5).

2. **Ca lỗi GraphRAG thất bại (hoặc cần cải thiện):**
   - *Question ID & Câu hỏi:* `G01` — *"Who was the CEO of Hugging Face in 2023?"*
   - *Nguyên nhân:* Ở câu hỏi Factoid này, bước trích xuất quan hệ NER/RE không phân loại vị trí "CEO" thành một cạnh quan hệ cụ thể trong allowlist 8 quan hệ (mà chỉ có `LEADS` hoặc thuộc tính), dẫn đến đồ thị thiếu cạnh trực tiếp kết nối `(Clément Delangue)-[:LEADS]->(Hugging Face)`. Trong khi đó, Flat RAG thực hiện Vector Search lấy ngay được chunk văn bản gốc chứa câu chữ *"Clément Delangue, CEO and co-founder of Hugging Face"* và trả lời hoàn hảo (5/5).
   - *Đề xuất khắc phục:* Áp dụng **Hybrid Retrieval Router** (Bonus A): Với câu hỏi Factoid, ưu tiên Vector Chunks retrieval; đồng thời bổ sung quan hệ `LEADS` và thuộc tính `role` chi tiết trong bước trích xuất Schema.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent
> **Trade-offs, Agent Control & Scale 350MB:** 
> - So sánh sự đánh đổi giữa GraphRAG vs Flat RAG về Latency, Token và Indexing Overhead.
> - Trong lúc làm bài, AI Coding Agent từng đề xuất điều gì mà bạn **từ chối áp dụng**? Tại sao?
> - Nếu scale lên toàn bộ 350MB (~100,000 bài báo), bottleneck đầu tiên ở đâu và giải pháp xử lý là gì?

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency:**
  - *Flat RAG:* Rẻ, nhanh, chi phí Indexing thấp (chỉ cần chạy Embedding model 1 lần). Rất mạnh ở câu hỏi Factoid / Single-document, nhưng mù mờ trước các câu hỏi Multi-hop và Cross-document liên kết nhiều thực thể.
  - *GraphRAG:* Chi phí Indexing cao (gọi LLM để trích xuất Triples, chạy Entity Resolution, Ingest vào Neo4j). Đổi lại, mang lại khả năng suy luận đa chặng chính xác, truy xuất nguồn gốc (provenance) rõ ràng và triệt tiêu hallucination trên các quan hệ phức tạp.
- **Quyết định từ chối AI Coding Agent:**
  1. *Từ chối thuật toán Pairwise Cosine $O(N^2)$:* Agent từng đề xuất so khớp từng cặp embedding trên toàn bộ dataset để làm Near-dedup và Entity resolution. Tôi đã từ chối và yêu cầu chuyển sang **FAISS FlatIP ANN Index** để giảm độ phức tạp xuống $O(N \log N)$ tránh tràn RAM.
  2. *Từ chối gộp thực thể chỉ dựa trên Cosine similarity $> 0.85$:* Từ chối vì gây lỗi gộp `GPT-3` vào `GPT-4` và `Sam Altman` vào `Steve Altman`. Bắt buộc bổ sung **Lexical Guard** nhiều tầng.
  3. *Từ chối chạy Cypher insert từng dòng:* Yêu cầu bắt buộc dùng cú pháp `UNWIND $rows AS row` theo batch 1000 records để đảm bảo hiệu năng ingestion.
- **Giải pháp scale 350MB:**
  - *Bottleneck 1 (LLM Extraction Throughput & Cost):* Áp dụng **Asynchronous Worker Queue** (Ray / Celery), kết hợp trích xuất phân tầng (sử dụng Local SLM như LLaMA-3-8B cho 80% văn bản thông thường và LLM lớn cho các bài báo phức tạp).
  - *Bottleneck 2 (Entity Resolution trên hàng triệu thực thể):* Sử dụng cơ chế **Blocking / Partitioning** (phân cụm theo Entity Type và chữ cái đầu) trước khi so khớp vector để tránh so khớp toàn cục.
  - *Bottleneck 3 (Super-node & Graph Traversal Explosion):* Áp dụng **Community Detection (Leiden Algorithm)** để tóm tắt trước các cụm đồ thị (Graph Summaries), kết hợp **Temporal Edge Indexing** và **Dynamic Degree Capping** ($N \le 50$).

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()` | Giảm thiểu False Edges khi xử lý đại từ trong văn bản kỹ thuật. |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Giữ cho đồ thị tri thức sạch, chuẩn hóa, không bị ô nhiễm bởi các relation tự phát. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | Tăng tốc độ nạp dữ liệu vào Neo4j gấp 20–50 lần so với insert đơn lẻ. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `UF` | Gộp hiệu quả các thực thể đồng nghĩa (`Microsoft Corp` → `Microsoft`), giữ 100% bản thể độc lập. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()` | Khống chế node trung tâm (`Google`, `Microsoft`) không làm nổ context window ($N \le 50$). |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()` | Đánh giá khách quan, định lượng hóa 3 tiêu chí: Comprehensiveness, Faithfulness, Multi-hop. |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Lỗi rate-limit TPM (Tokens Per Minute) khi gửi toàn bộ context đồ thị lớn qua LLM-as-a-Judge, kèm theo hiện tượng False Merges giữa các phiên bản công nghệ có vector tương đồng cao (`GPT-3` vs `GPT-4`).
- **Cách bạn đã xử lý thành công:** 
  1. Thiết kế **Lexical Guard** với Regex kiểm tra xung đột chữ số và phân biệt tên riêng.
  2. Tối ưu hóa dung lượng Linearized Subgraph Context ($\le 3,500$ chars) và thêm cơ chế retry với exponential backoff.
  3. Xây dựng **Dual-Mode Graph Engine** hỗ trợ chạy mượt mà cả với Neo4j AuraDB từ xa và In-Memory Cypher Engine cục bộ.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** Hệ thống Trợ lý Pháp lý & Phân tích Chuỗi Sở hữu Doanh nghiệp (Enterprise & Legal Intelligence GraphRAG).
- **Đặc thù bài toán & Lý do chọn giải pháp:** Bài toán tra cứu pháp lý và M&A doanh nghiệp đòi hỏi suy luận xuyên suốt nhiều văn bản (cross-doc) qua nhiều tầng công ty con, cổ đông và người đại diện pháp luật (multi-hop). Flat RAG hoàn toàn thất bại trong việc truy vết đường đi sở hữu, do đó **Hybrid GraphRAG** là kiến trúc bắt buộc.
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Company`, `Person`, `LegalDocument`, `Jurisdiction`, `Asset`
  - Relations: `OWNS_SHARES`, `REPRESENTS`, `ACQUIRED`, `SUBSIDIARY_OF`, `REGULATED_BY`
- **Chiến lược xử lý Super-node & Entity Resolution:** 
  - Sử dụng Mã số Doanh nghiệp (Tax ID / Business ID) làm khóa định danh cứng (Hard ID) kết hợp Vector Matching cho tên công ty.
  - Phân tầng Super-node (các tập đoàn lớn như Vingroup, Viettel) bằng **Community Reports** cấp độ 1 và 2 để truy vấn tổng quan trước khi đi sâu vào chi tiết hợp đồng.

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | **5/5** | Nắm vững toàn bộ pipeline từ Preprocessing, Entity Resolution, Traversal đến Evaluation. |
| Khả năng kiểm soát AI Coding Agent | **5/5** | Chủ động kiểm soát schema, từ chối thuật toán $O(N^2)$, thiết lập Lexical Guard và UNWIND bulk. |
| Chất lượng đồ thị tri thức xây dựng | **5/5** | 100% edge có provenance (`source_chunk_id`, `published_date`, `evidence`), 0 invalid edge. |
| Khả năng phân tích và debug hệ thống | **5/5** | Xử lý triệt để rate limit, phân tích định lượng rõ nét ưu/nhược điểm của Flat RAG vs GraphRAG. |
