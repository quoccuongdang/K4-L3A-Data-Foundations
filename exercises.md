# Ngày 7 — Bài tập
## Nền tảng Dữ liệu: Embedding & Vector Store | Bài tập thực hành

---

## Phần 1 — Khởi động (Cá nhân)

### Bài tập 1.1 — Cosine Similarity (Độ tương tự Cosine) bằng ngôn ngữ đời thường

Không yêu cầu toán học — hãy giải thích về mặt khái niệm:

- Điều gì xảy ra khi hai đoạn văn bản có độ tương tự cosine cao?
- Đưa ra một ví dụ cụ thể về hai câu sẽ có độ tương tự CAO và hai câu sẽ có độ tương tự THẤP.
- Tại sao độ tương tự cosine lại được ưu tiên hơn khoảng cách Euclid (Euclidean distance) đối với text embeddings?

**Trả lời:**

1. **Khi hai đoạn văn bản có độ tương tự cosine cao**, điều đó có nghĩa là hai đoạn văn bản đó mang ý nghĩa ngữ nghĩa (semantic meaning) gần giống nhau — chúng nói về cùng một chủ đề, truyền tải cùng một ý tưởng, hoặc chứa thông tin tương đương, dù có thể được diễn đạt bằng những từ ngữ khác nhau. Về mặt hình học, hai vector embedding của chúng "chỉ cùng một hướng" trong không gian nhiều chiều.

2. **Ví dụ cụ thể:**
   - **Độ tương tự CAO:** *"Sinh viên cần đăng ký môn học trước khi kỳ học bắt đầu"* và *"Việc đăng ký các học phần phải được hoàn tất trước ngày khai giảng"* — cả hai câu đều nói về cùng một nội dung (đăng ký môn trước kỳ học) dù dùng từ khác nhau.
   - **Độ tương tự THẤP:** *"Sinh viên cần đăng ký môn học trước khi kỳ học bắt đầu"* và *"Thời tiết hôm nay nắng đẹp, nhiệt độ khoảng 30 độ C"* — hai câu hoàn toàn khác chủ đề, không liên quan về mặt ngữ nghĩa.

3. **Tại sao cosine similarity được ưu tiên hơn Euclidean distance đối với text embeddings:**
   - **Không phụ thuộc vào độ dài vector (magnitude):** Cosine similarity chỉ đo **góc** giữa hai vector, không quan tâm đến độ dài. Hai văn bản dài ngắn khác nhau vẫn có thể cùng ngữ nghĩa — cosine sẽ phản ánh đúng điều này, trong khi Euclidean distance bị ảnh hưởng bởi sự khác biệt về độ lớn (magnitude) của vector.
   - **Hiệu quả trong không gian nhiều chiều:** Trong không gian embedding chiều cao (thường 384–1536 chiều), Euclidean distance trở nên kém phân biệt (hiện tượng "curse of dimensionality" — lời nguyền chiều cao) vì khoảng cách giữa các điểm có xu hướng trở nên đồng đều. Cosine similarity vẫn giữ được khả năng phân biệt tốt vì nó so sánh hướng thay vì khoảng cách tuyệt đối.
   - **Giá trị trả về trực quan:** Cosine similarity luôn nằm trong khoảng [-1, 1] (với text embeddings thường là [0, 1]), dễ diễn giải hơn so với Euclidean distance có giá trị không bị giới hạn trên.

> **Ghi kết quả vào:** Báo cáo — Phần 1 (Khởi động)

---

### Bài tập 1.2 — Bài toán tính toán Chunking

- Một tài liệu có độ dài 10,000 ký tự. Bạn tiến hành chia nhỏ (chunk) với `chunk_size=500` (kích thước chunk), `overlap=50` (độ chồng chéo). Bạn dự kiến sẽ có bao nhiêu chunks?
- Công thức: `số lượng chunk = làm_tròn_lên((độ_dài_tài_liệu - độ_chồng_chéo) / (kích_thước_chunk - độ_chồng_chéo))`
- Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk sẽ thay đổi như thế nào? Tại sao bạn lại muốn tăng độ chồng chéo?

**Trả lời:**

1. **Với `chunk_size=500`, `overlap=50`:**

   Áp dụng công thức:
   ```
   số lượng chunk = ceil((10000 - 50) / (500 - 50))
                  = ceil(9950 / 450)
                  = ceil(22.11)
                  = 23 chunks
   ```
   → Dự kiến sẽ có **23 chunks**.

2. **Khi tăng `overlap` lên 100:**

   ```
   số lượng chunk = ceil((10000 - 100) / (500 - 100))
                  = ceil(9900 / 400)
                  = ceil(24.75)
                  = 25 chunks
   ```
   → Số lượng chunk **tăng từ 23 lên 25 chunks** (tăng thêm 2 chunks).

3. **Tại sao muốn tăng độ chồng chéo (overlap)?**
   - **Bảo toàn ngữ cảnh tại ranh giới chunk:** Khi overlap lớn hơn, phần nội dung giao nhau giữa hai chunk liền kề nhiều hơn, giúp tránh tình trạng một câu hoặc một ý bị cắt đôi và mất ngữ cảnh tại điểm nối.
   - **Tăng khả năng truy xuất (retrieval recall):** Nếu thông tin quan trọng nằm ngay tại ranh giới giữa hai chunk, overlap lớn hơn đảm bảo thông tin đó xuất hiện trọn vẹn trong ít nhất một chunk, giúp hệ thống tìm kiếm dễ tìm thấy đúng chunk liên quan hơn.
   - **Đánh đổi (trade-off):** Overlap lớn hơn đồng nghĩa với nhiều chunk hơn → tốn nhiều bộ nhớ hơn và tăng chi phí tính toán embedding. Cần cân bằng giữa chất lượng truy xuất và chi phí lưu trữ/tính toán.

> **Ghi kết quả vào:** Báo cáo — Phần 1 (Khởi động)

---

## Phần 2 — Lập trình cốt lõi (Cá nhân)

Hoàn thành tất cả các TODOs trong `src/chunking.py`, `src/store.py`, và `src/agent.py`. `Document` dataclass và `FixedSizeChunker` đã được triển khai sẵn làm ví dụ — hãy đọc kỹ để hiểu cấu trúc trước khi lập trình phần còn lại.

Chạy `pytest tests/` để kiểm tra tiến độ.

### Danh sách cần làm (Checklist)
- [x] `Document` dataclass — ĐÃ TRIỂN KHAI SẴN
- [x] `FixedSizeChunker` — ĐÃ TRIỂN KHAI SẴN
- [ ] `SentenceChunker` — tách dựa trên ranh giới câu, nhóm lại thành các chunks
- [ ] `RecursiveChunker` — thử nghiệm các dấu phân cách (separators) theo thứ tự, thực hiện đệ quy trên các đoạn có kích thước quá lớn
- [ ] `compute_similarity` — công thức tính độ tương tự cosine kèm cơ chế bảo vệ chia cho 0
- [ ] `ChunkingStrategyComparator` — gọi cả ba chiến lược, tính toán các chỉ số thống kê
- [ ] `EmbeddingStore.__init__` — khởi tạo store (lưu trữ trong bộ nhớ hoặc ChromaDB)
- [ ] `EmbeddingStore.add_documents` — nhúng (embed) và lưu trữ từng tài liệu
- [ ] `EmbeddingStore.search` — nhúng truy vấn, xếp hạng theo tích vô hướng (dot product)
- [ ] `EmbeddingStore.get_collection_size` — trả về số lượng
- [ ] `EmbeddingStore.search_with_filter` — lọc theo siêu dữ liệu (metadata), sau đó tìm kiếm
- [ ] `EmbeddingStore.delete_document` — xóa tất cả các chunks của một doc_id
- [ ] `KnowledgeBaseAgent.answer` — truy xuất (retrieve) + tạo prompt + gọi LLM

> **Nộp code:** thư mục `src/`
> **Ghi lại hướng tiếp cận vào:** Báo cáo — Phần 4 (Hướng tiếp cận của tôi)

---

## Phần 3 — So Sánh Chiến Lược Truy Xuất (Nhóm)

### Bài tập 3.0 — Chuẩn Bị Tài Liệu (Giờ đầu tiên)

Mỗi nhóm chọn một chủ đề (domain) và chuẩn bị bộ tài liệu:

**Bước 1 — Chọn chủ đề:** FAQ (Câu hỏi thường gặp), SOP (Quy trình chuẩn), chính sách, tài liệu kỹ thuật, công thức nấu ăn, luật, y tế, v.v.

**Bước 2 — Thu thập 5-10 tài liệu.** Chỉ dùng nguồn công khai hoặc nguồn nhóm có quyền sử dụng; lưu dưới dạng `.txt` hoặc `.md` vào thư mục `data/`.

**Quy tắc dữ liệu bắt buộc:**
- Không đưa dữ liệu cá nhân, thông tin đăng nhập, hồ sơ nội bộ hoặc nội dung có quyền sử dụng không rõ ràng vào repo.
- Với mỗi tài liệu, ghi `source_url`, `retrieved_at` (ngày lấy) và `document_version` hoặc ngày hiệu lực nếu nguồn có nêu.
- Đưa ba trường trên vào siêu dữ liệu (metadata) khi nạp (ingest); chúng giúp kiểm tra độ mới và truy vết câu trả lời.

> **Mẹo chuyển PDF sang Markdown:**
> - `pip install marker-pdf` → `marker_single input.pdf output/` (chất lượng cao, giữ cấu trúc)
> - `pip install pymupdf4llm` → `pymupdf4llm.to_markdown("input.pdf")` (nhanh, đơn giản)
> - Hoặc sao chép-dán (copy-paste) nội dung từ PDF/web vào file `.txt`

Ghi vào bảng:

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bước 3 — Thiết kế cấu trúc metadata (metadata schema):** Mỗi tài liệu cần `source_url`, `retrieved_at`, `document_version` và ít nhất 2 trường hữu ích cho việc truy xuất (ví dụ: `audience`, `department`, `category`, `language`, `difficulty`).

> **Ghi kết quả vào:** Báo cáo — Phần 2 (Lựa chọn tài liệu)

---

### Bài tập 3.1 — Thiết Kế Chiến Lược Truy Xuất (Mỗi người thử riêng)

Mỗi thành viên **tự chọn chiến lược riêng** để thử nghiệm trên cùng bộ tài liệu của nhóm.

**Bước 1 — Đường cơ sở (Baseline):** Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu. Ghi lại kết quả.

**Trả lời — Kết quả Baseline (chunk_size=500):**

| Tài liệu | Chiến lược | Số chunk | Avg length |
|-----------|-----------|----------|------------|
| 01-academic-regulations.md (21,107 chars) | fixed_size | 53 | 496 |
| | by_sentences | 73 | 288 |
| | recursive | 68 | 309 |
| 02-fap-and-academic-procedures.md (4,462 chars) | fixed_size | 11 | 497 |
| | by_sentences | 9 | 493 |
| | recursive | 13 | 341 |
| 03-tuition-hcm.md (2,930 chars) | fixed_size | 8 | 454 |
| | by_sentences | 2 | 1464 |
| | recursive | 7 | 417 |

**Nhận xét baseline:**
- `fixed_size` luôn cho chunk có kích thước đều đặn (~496-497), nhưng có thể cắt giữa câu/ý.
- `by_sentences` có thể tạo chunk rất dài (1,464 ký tự với tài liệu có ít dấu chấm câu chuẩn) hoặc rất ngắn.
- `recursive` cân bằng tốt giữa kích thước và cấu trúc ngữ nghĩa.

**Bước 2 — Chọn hoặc thiết kế chiến lược của bạn:**
- Dùng 1 trong 3 chiến lược có sẵn (built-in strategies) với tham số tối ưu, HOẶC
- Thiết kế chiến lược tùy chỉnh cho chủ đề của bạn (ví dụ: chia nhỏ theo cặp Câu hỏi-Đáp án, theo các phần (sections), theo tiêu đề (headers))
- Mỗi thành viên nên thử một chiến lược **khác nhau** để có cơ sở so sánh

**Trả lời — Chiến lược tùy chỉnh: `HeaderChunker`**

```python
class HeaderChunker:
    """Chiến lược chia nhỏ tùy chỉnh cho tài liệu đại học (Markdown).

    Lý do thiết kế: Tài liệu từ daihoc.fpt.edu.vn có cấu trúc rõ ràng theo
    heading Markdown (##, ###). Chia theo heading giữ nguyên ngữ cảnh của từng
    mục (ví dụ: 1 điều khoản, 1 câu hỏi FAQ, 1 phần hướng dẫn), giúp mỗi
    chunk là một đơn vị nội dung hoàn chỉnh — phù hợp hơn cho retrieval.
    """

    def __init__(self, max_chunk_size: int = 800):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Split on markdown headings (##, ###)
        sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]

        chunks = []
        for section in sections:
            if len(section) <= self.max_chunk_size:
                chunks.append(section)
            else:
                # If section too long, split by paragraphs
                paragraphs = section.split('\n\n')
                current = ''
                for para in paragraphs:
                    if current and len(current) + len(para) + 2 > self.max_chunk_size:
                        chunks.append(current.strip())
                        current = para
                    else:
                        current = current + '\n\n' + para if current else para
                if current.strip():
                    chunks.append(current.strip())
        return chunks
```

**Bước 3 — So sánh:** So sánh chiến lược tùy chỉnh/được tinh chỉnh (custom/tuned strategy) với đường cơ sở (baseline) trên cùng tài liệu.

**Trả lời — So sánh HeaderChunker vs Baseline:**

| Tài liệu | fixed_size | by_sentences | recursive | **header_based** |
|-----------|-----------|-------------|-----------|-----------------|
| 01-academic-regulations.md | 53 chunks (496) | 73 chunks (288) | 68 chunks (309) | **31 chunks (679)** |
| 02-fap-and-academic-procedures.md | 11 chunks (497) | 9 chunks (493) | 13 chunks (341) | **12 chunks (370)** |
| 03-tuition-hcm.md | 8 chunks (454) | 2 chunks (1464) | 7 chunks (417) | **4 chunks (731)** |

**Phân tích:**
- `header_based` tạo ít chunk hơn nhưng mỗi chunk có **ngữ cảnh hoàn chỉnh** (một section/mục trọn vẹn).
- Với tài liệu quy chế (01), HeaderChunker giảm từ 53-73 chunk xuống chỉ 31 chunk mà không mất thông tin — mỗi chunk tương ứng một điều khoản.
- Nhược điểm: avg_length lớn hơn (679-731), có thể gây nhiễu nếu chunk chứa nhiều ý phụ. Cần cân nhắc `max_chunk_size` phù hợp.

> **Ghi kết quả vào:** Báo cáo — Phần 3 (Chiến lược chia nhỏ - Chunking Strategy)

---

### Bài tập 3.2 — Chuẩn Bị Câu Hỏi Đánh Giá (Benchmark Queries)

Mỗi nhóm viết **đúng 5 câu hỏi đánh giá** kèm theo **câu trả lời chuẩn (gold answers)**.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

**Yêu cầu:**
- Câu hỏi phải đa dạng (không hỏi 5 câu có nội dung/cấu trúc giống hệt nhau)
- Câu trả lời chuẩn phải cụ thể và có thể kiểm chứng (verify) từ tài liệu
- Ít nhất 1 câu hỏi yêu cầu lọc bằng metadata (metadata filtering) để trả lời tốt

> **Ghi kết quả vào:** Báo cáo — Phần 6 (Kết quả — Câu hỏi đánh giá & Câu trả lời chuẩn)

---

### Bài tập 3.3 — Dự Đoán Độ Tương Tự Cosine (Cá nhân)

Gọi hàm `compute_similarity()` trên 5 cặp câu. **Trước khi chạy**, hãy dự đoán xem cặp câu nào sẽ có độ tương tự cao nhất/thấp nhất. Ghi lại các dự đoán của bạn và kết quả thực tế. Suy ngẫm xem điều gì khiến bạn ngạc nhiên nhất.

**Trả lời:**

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|-----|-------|-------|---------|--------------|-------|
| 1 | "Sinh vien can dang ky mon hoc truoc khi ky hoc bat dau" | "Viec dang ky cac hoc phan phai duoc hoan tat truoc ngay khai giang" | Cao | 0.3207 | Đúng (cao nhất) |
| 2 | "Hoc phi cua Dai hoc FPT la bao nhieu?" | "Chi phi hoc tap tai truong FPT nhu the nao?" | Cao | -0.1116 | Sai |
| 3 | "Sinh vien can dang ky mon hoc truoc khi ky hoc bat dau" | "Thoi tiet hom nay nang dep, nhiet do khoang 30 do C" | Thấp | 0.0182 | Đúng |
| 4 | "Quy che dao tao quy dinh ve diem GPA tot nghiep" | "Dieu kien tot nghiep la phai dat GPA toi thieu" | Cao | -0.1500 | Sai |
| 5 | "Chuong trinh trao doi quoc te giup sinh vien mo rong tam nhin" | "Hoc phi tai campus Ho Chi Minh duoc cong bo hang nam" | Thấp | -0.0910 | Đúng |

**Dự đoán cao nhất:** Cặp 1 (cùng nói về đăng ký môn) → Đúng (0.3207)
**Dự đoán thấp nhất:** Cặp 3 (đăng ký môn vs thời tiết) → Đúng (0.0182, gần 0)

**Điều bất ngờ nhất:** Cặp 2 và cặp 4 — hai cặp câu mà con người đánh giá là rất tương đồng (cùng hỏi về học phí, cùng nói về GPA tốt nghiệp) nhưng mock embedder cho điểm **âm** (-0.1116 và -0.1500). Điều này cho thấy **mock embedder (dùng hash MD5)** chỉ tạo vector giả lập dựa trên chuỗi ký tự, **không nắm bắt được ngữ nghĩa thực sự**. Trong thực tế, cần dùng mô hình embedding thật (sentence-transformers, OpenAI, Gemini) để có kết quả phản ánh đúng mối quan hệ ngữ nghĩa.

> **Ghi kết quả vào:** Báo cáo — Phần 5 (Dự đoán độ tương tự)

---

### Bài tập 3.4 — Chạy Đánh Giá & So Sánh Trong Nhóm

**Bước 1:** Mỗi thành viên chạy 5 câu hỏi đánh giá với chiến lược riêng. Ghi lại kết quả top-3 cho mỗi câu hỏi.

**Bước 2:** So sánh kết quả trong nhóm:
- Chiến lược nào cho việc truy xuất tốt nhất? Tại sao?
- Có câu hỏi nào mà chiến lược A tốt hơn B nhưng lại ngược lại ở câu hỏi khác không?
- Lọc bằng metadata (Metadata filtering) có giúp ích không?

**Bước 3:** Thảo luận và rút ra bài học — chuẩn bị cho phần demo (thuyết trình) với các nhóm khác.

> **Ghi kết quả vào:** Báo cáo — Phần 6 (Kết quả)
> **Gợi ý đánh giá:** xem danh sách kiểm tra ngắn trong `README.md` mục **Cách Tự Đánh Giá Kết Quả Retrieval** hoặc chi tiết hơn trong file `docs/EVALUATION.md`.

---

### Bài tập 3.5 — Phân Tích Lỗi (Failure Analysis)

Tìm ít nhất **1 trường hợp lỗi (failure case)** trong quá trình so sánh. Mô tả:
- Câu hỏi nào mà quá trình truy xuất gặp thất bại?
- Tại sao? (do chunk quá nhỏ/quá lớn, thiếu metadata, câu hỏi mơ hồ, v.v.)
- Đề xuất cải thiện?

**Trả lời — Trường hợp lỗi 1:**

- **Câu hỏi:** "Dieu kien tot nghiep dai hoc FPT la gi?"
- **Kết quả truy xuất:** Top-1 trả về `06-campus-facilities-hcm.md` (category=campus_services, score=0.2646) — đây là tài liệu về **cơ sở vật chất campus**, hoàn toàn không liên quan đến điều kiện tốt nghiệp.
- **Chunk đúng nên trả về:** `01-academic-regulations.md` (quy chế đào tạo — chứa thông tin về GPA, tín chỉ tốt nghiệp)
- **Nguyên nhân lỗi:**
  1. **Mock embedder không hiểu ngữ nghĩa:** Hash MD5 tạo vector giả lập dựa trên ký tự, không phân biệt được "tốt nghiệp" với "cơ sở vật chất".
  2. **Chunk quá lớn (toàn bộ document):** Khi chưa chunk mà lưu cả document, vector embedding bị "pha loãng" bởi metadata YAML front matter + nội dung hỗn tạp, khiến similarity score không phản ánh đúng.
  3. **Thiếu pre-filtering:** Nếu dùng `search_with_filter(metadata_filter={"category": "academic_regulation"})`, kết quả sẽ chính xác hơn nhiều.
- **Đề xuất cải thiện:**
  - Dùng mô hình embedding thật (sentence-transformers hoặc Gemini API) thay vì mock.
  - Chia nhỏ tài liệu bằng `HeaderChunker` hoặc `RecursiveChunker` trước khi lưu vào store, thay vì lưu toàn bộ document.
  - Tận dụng metadata filtering (`category`, `department`) để thu hẹp phạm vi tìm kiếm.

**Trả lời — Trường hợp lỗi 2:**

- **Câu hỏi:** "Hoc phi tai campus Ho Chi Minh la bao nhieu?"
- **Kết quả truy xuất:** Top-1 trả về `05-student-services-hcm.md` (category=student_services, score=0.0868) — tài liệu về dịch vụ sinh viên/liên hệ, không phải về học phí.
- **Chunk đúng nên trả về:** `03-tuition-hcm.md` (học phí HCM)
- **Nguyên nhân lỗi:**
  1. `03-tuition-hcm.md` có score rất thấp (không nằm trong top-3) dù chứa đúng thông tin cần tìm.
  2. Mock embedder không nhận ra mối liên hệ giữa "hoc phi" và nội dung bảng học phí trong tài liệu 03.
  3. **Retrieval precision thấp:** Score cao nhất chỉ 0.0868 — tất cả kết quả đều gần 0, cho thấy hệ thống không tìm được chunk thực sự liên quan.
- **Đề xuất cải thiện:**
  - Sử dụng embedding thật sẽ giải quyết phần lớn vấn đề này.
  - Thiết kế prompt query tốt hơn hoặc dùng query expansion.
  - Kết hợp keyword search (BM25) với semantic search để tăng recall.

> **Ghi kết quả vào:** Báo cáo — Phần 7 (Những gì tôi học được)
> **Gợi ý:** phân tích lỗi nên tham chiếu từ các góc nhìn như độ chính xác (precision), tính mạch lạc của chunk (chunk coherence), tính hữu dụng của metadata, và chất lượng thông tin nền (grounding quality).

---

## Danh Sách Kiểm Tra Nộp Bài (Submission Checklist)

- [x] Vượt qua tất cả các bài kiểm thử (tests): `pytest tests/ -v` — **42/42 PASSED**
- [x] Cập nhật thư mục `src/` (cá nhân)
- [ ] Hoàn thành báo cáo nhóm (`report/REPORT_NHOM.md` — 1 file/nhóm) — *Chờ nhóm chốt benchmark*
- [x] Hoàn thành báo cáo cá nhân (`report/REPORT_CANHAN.md` — 1 file/sinh viên)
