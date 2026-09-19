# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Đặng Quốc Cường
**Nhóm:** G36
**Ngày:** 2026-09-19

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Khi hai đoạn văn bản có độ tương tự cosine cao, điều đó có nghĩa là hai vector embedding của chúng "chỉ cùng một hướng" trong không gian nhiều chiều — tức là chúng mang ý nghĩa ngữ nghĩa gần giống nhau, dù có thể được diễn đạt bằng những từ ngữ khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên cần đăng ký môn học trước khi kỳ học bắt đầu"
- Câu B: "Việc đăng ký các học phần phải được hoàn tất trước ngày khai giảng"
- Tại sao tương đồng: Cả hai câu đều nói về cùng một nội dung (đăng ký môn trước kỳ học) dù dùng từ khác nhau ("môn học" vs "học phần", "kỳ học bắt đầu" vs "ngày khai giảng").

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên cần đăng ký môn học trước khi kỳ học bắt đầu"
- Câu B: "Thời tiết hôm nay nắng đẹp, nhiệt độ khoảng 30 độ C"
- Tại sao khác: Hai câu hoàn toàn khác chủ đề — một câu về quy trình đăng ký học, một câu về thời tiết — không có sự liên quan về mặt ngữ nghĩa.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine similarity chỉ đo **góc** giữa hai vector, không phụ thuộc vào độ lớn (magnitude), nên hai văn bản dài ngắn khác nhau nhưng cùng ngữ nghĩa vẫn được đánh giá đúng. Trong không gian embedding chiều cao (384–1536 chiều), Euclidean distance bị ảnh hưởng bởi "curse of dimensionality" khiến khoảng cách giữa các điểm trở nên đồng đều và kém phân biệt, trong khi cosine similarity vẫn giữ được khả năng phân biệt tốt. Ngoài ra, giá trị cosine nằm trong khoảng [-1, 1] nên dễ diễn giải hơn.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Phép tính:* `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> *Đáp án:* **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25 chunks` — tăng từ 23 lên 25 chunks. Overlap lớn hơn giúp bảo toàn ngữ cảnh tại ranh giới giữa các chunk (tránh cắt đứt ý giữa chừng), tăng khả năng truy xuất đúng chunk liên quan, nhưng đánh đổi bằng việc tốn nhiều bộ nhớ và chi phí tính toán embedding hơn.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng biểu thức chính quy `re.split(r'(?<=[.!?])(?:\s|\n)', text)` với **lookbehind assertion** để tách câu dựa trên dấu kết thúc câu (`.`, `!`, `?`) theo sau bởi khoảng trắng hoặc xuống dòng. Sau đó nhóm các câu lại theo `max_sentences_per_chunk` bằng vòng lặp `range(0, len(sentences), max_sentences_per_chunk)` và nối bằng dấu cách. Xử lý edge case: text rỗng trả `[]`, câu rỗng sau khi strip được loại bỏ, và `max_sentences_per_chunk` được bảo vệ tối thiểu là 1 ngay từ `__init__`.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán đệ quy thử từng separator theo thứ tự ưu tiên (`\n\n` → `\n` → `. ` → ` ` → `""`). Với mỗi separator, `split()` text thành các phần rồi **merge các phần nhỏ** lại sao cho mỗi chunk ≤ `chunk_size`. Nếu phần nào vẫn quá lớn thì đệ quy xuống separator tiếp theo. Base case: (1) text đã ≤ `chunk_size` → trả `[text]`, (2) hết separator hoặc gặp separator rỗng `""` → hard-split theo `chunk_size`. Nếu separator không chia được text (chỉ 1 phần) → nhảy sang separator tiếp.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Mỗi document được chuyển thành record dict gồm `{id, content, metadata, embedding}` qua hàm `_make_record()`, trong đó embedding được tính bằng `self._embedding_fn(doc.content)`. Các record được append vào `self._store` (list in-memory). Khi search, embed query rồi tính **dot product** giữa query embedding với mỗi record embedding bằng hàm `_dot()`, sắp xếp giảm dần theo score và trả về top_k kết quả. Hỗ trợ cả ChromaDB backend (nếu có cài) thông qua `collection.add()` và `collection.query()`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` thực hiện **lọc trước, tìm kiếm sau** (pre-filter): duyệt qua `self._store`, chỉ giữ lại các record có metadata khớp **tất cả** key-value trong `metadata_filter`, rồi chạy `_search_records()` trên danh sách đã lọc. Nếu `metadata_filter=None` thì gọi thẳng `self.search()`. `delete_document` dùng list comprehension lọc bỏ tất cả record có `id == doc_id`, so sánh length trước/sau để trả `True/False`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Luồng RAG 3 bước: (1) Gọi `self.store.search(question, top_k)` để lấy top-k chunks liên quan nhất, (2) Ghép các chunk thành context dạng đánh số `[1] chunk_content...`, rồi xây dựng prompt gồm instruction + context + question, (3) Gọi `self.llm_fn(prompt)` và trả về kết quả. Prompt có hướng dẫn LLM chỉ trả lời dựa trên context được cung cấp.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\ADMIN\Desktop\AI in Action\K4-L3A-Data-Foundations
plugins: anyio-4.14.0, langsmith-0.8.17
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.19s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Sinh vien can dang ky mon hoc truoc khi ky hoc bat dau" | "Viec dang ky cac hoc phan phai duoc hoan tat truoc ngay khai giang" | cao | 0.3207 | Đúng (cao nhất) |
| 2 | "Hoc phi cua Dai hoc FPT la bao nhieu?" | "Chi phi hoc tap tai truong FPT nhu the nao?" | cao | -0.1116 | Sai |
| 3 | "Sinh vien can dang ky mon hoc truoc khi ky hoc bat dau" | "Thoi tiet hom nay nang dep, nhiet do khoang 30 do C" | thấp | 0.0182 | Đúng |
| 4 | "Quy che dao tao quy dinh ve diem GPA tot nghiep" | "Dieu kien tot nghiep la phai dat GPA toi thieu" | cao | -0.1500 | Sai |
| 5 | "Chuong trinh trao doi quoc te giup sinh vien mo rong tam nhin" | "Hoc phi tai campus Ho Chi Minh duoc cong bo hang nam" | thấp | -0.0910 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là **cặp 2 và cặp 4**: hai cặp câu mà con người đánh giá là rất tương đồng về ngữ nghĩa (cùng hỏi về học phí, cùng nói về GPA tốt nghiệp) nhưng mock embedder lại cho điểm âm (-0.1116 và -0.1500). Điều này cho thấy **mock embedder** (dùng hash MD5 để tạo vector) **không nắm bắt được ngữ nghĩa thực sự** của văn bản — nó chỉ tạo vector giả lập dựa trên chuỗi ký tự, không hiểu nghĩa. Trong thực tế, cần dùng mô hình embedding thật (như `sentence-transformers` hoặc OpenAI embeddings) để có kết quả phản ánh đúng mối quan hệ ngữ nghĩa giữa các câu.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên cần đáp ứng đầy đủ những điều kiện nào để tham gia OJT? | Yêu cầu đọc tài liệu OJT và tham gia đầy đủ buổi Orientation | 0.8029 | Có, nhưng thiếu ngưỡng tín chỉ | Agent nêu rõ việc đọc tài liệu, tham gia Orientation và đối chiếu kết quả học tập/tài chính, nhưng thiếu ngưỡng 90% tín chỉ chuyên môn. |
| 2 | Học phí mỗi học kỳ năm 2026 của ngành Trí tuệ nhân tạo tại TP.HCM là bao nhiêu cho KV1 và các khu vực khác? | Phạm vi học phí áp dụng cho tân sinh viên khóa K22 năm 2026 | 0.7775 | Có một phần; giá tiền ở top-2 | Agent trả lời chính xác: KV1 là 22.120.000 đồng/kỳ, các khu vực khác là 31.600.000 đồng/kỳ. |
| 3 | Hạn nộp hồ sơ học bổng năm 2026 là khi nào và GPA tối thiểu để duy trì học bổng là bao nhiêu? | Hạn nộp hồ sơ học bổng là ngày 15/5/2026 | 0.7291 | Có; GPA duy trì ở top-2 | Agent trả lời đúng hạn chót 15/5/2026 và điều kiện duy trì học bổng là GPA ≥ 7.0/10. |
| 4 | Trên FAP, sinh viên gửi và theo dõi đơn online như thế nào, đồng thời xem báo cáo điểm danh ở đâu? | Hướng dẫn vào mục Báo cáo và xem báo cáo điểm danh | 0.7843 | Có; mục gửi đơn ở top-4 | Agent hướng dẫn đầy đủ: chọn Gửi Đơn, theo dõi tại Xem Đơn, và vào Báo cáo để xem điểm danh. |
| 5 | Sinh viên gặp vấn đề về thủ tục hành chính hoặc đời sống trong quá trình học tại campus TP.HCM thì liên hệ đơn vị nào, hotline và phòng bao nhiêu? | Phạm vi hỗ trợ thủ tục hành chính, hotline và phòng làm việc | 0.8280 | Có, đầy đủ | Agent trả lời chính xác: Phòng Dịch vụ Sinh viên, hotline 028 7300 5585, tại phòng 202 campus TP.HCM. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5. Tuy nhiên chỉ 4 / 5 câu có **đầy đủ** tất cả bằng chứng trong top-5; Q1 là failure case.

**Kết quả macro theo 7 metric của nhóm:** Recall@1 = 53,3%; Recall@5 = 86,7%; MRR = 100%; nDCG@5 = 76,1%; Full Evidence@5 = 80%; Faithfulness / Agent Accuracy = 73,3%; Audience Match Rate = 100%.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua quá trình chạy benchmark 5 câu hỏi và so sánh giữa các chiến lược trong nhóm, bài học lớn nhất mà tôi rút ra là: **chiến lược chunking dựa trên cấu trúc ngữ nghĩa (Semantic / Header-based) kết hợp với Metadata Filtering mang lại chất lượng truy xuất vượt trội hơn hẳn so với Fixed-size chunking thuần túy**. Đặc biệt ở trường hợp thất bại của câu Q1 (thông tin điều kiện OJT bị phân tán giữa quy chế đào tạo chung và thông báo OJT cụ thể), tôi nhận thấy một hệ thống RAG thực tế không thể chỉ dựa vào một câu query vector đơn lẻ mà cần phải kết hợp kỹ thuật Multi-query hoặc Hybrid Search (kết hợp từ khóa BM25 và Vector) để bao phủ trọn vẹn bằng chứng cần thiết cho LLM.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
