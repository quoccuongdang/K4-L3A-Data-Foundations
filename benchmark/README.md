# FPTU RAG Benchmark

Bộ benchmark dùng 5 câu hỏi chung của nhóm và 7 metric:

1. `Recall@1`
2. `Recall@5`
3. `MRR`
4. `nDCG@5`
5. `Full Evidence@5`
6. `Faithfulness / Agent Accuracy`
7. `Audience Match Rate`

## Chạy benchmark

Chạy đầy đủ retrieval và sinh câu trả lời bằng Gemini:

```bash
python bench.py
```

Chỉ chấm retrieval, không gọi LLM:

```bash
python bench.py --retrieval-only
```

Chạy A/B với metadata pre-filter:

```bash
python bench.py
python bench.py --audience-filter
```

Có thể chọn embedding backend bằng `--provider gemini|local|openai|mock`. Không dùng `mock` để kết luận chất lượng vì mock embedder không mã hóa ngữ nghĩa.

## Tệp dữ liệu và kết quả

- `gold_queries.json`: query, Gold Answer, đơn vị bằng chứng và tiêu chí từ khóa của Agent.
- `results.json`: kết quả có cấu trúc của lần chạy gần nhất.
- `../ket_qua_benchmark.txt`: báo cáo đọc nhanh, gồm bảng macro và top-5 từng query.

## Quy ước chấm

Một chunk chỉ được xem là liên quan khi nội dung của nó khớp ít nhất một **đơn vị bằng chứng**, không chấm chỉ dựa trên `doc_id`. Quy ước này tránh trường hợp lấy đúng tài liệu nhưng sai đoạn trả lời.

- `Recall@k`: số đơn vị bằng chứng khác nhau có trong top-k chia tổng số đơn vị bằng chứng cần có.
- `MRR`: nghịch đảo thứ hạng của chunk đầu tiên chứa bằng chứng.
- `nDCG@5`: relevance grade của chunk bằng số đơn vị bằng chứng chunk đó chứa.
- `Full Evidence@5`: top-5 phải bao phủ tất cả đơn vị bằng chứng.
- `Faithfulness / Agent Accuracy`: tỷ lệ tiêu chí từ khóa gold xuất hiện trong câu trả lời. Đây là phép kiểm tra deterministic theo thống nhất nhóm, chưa đánh giá được mọi dạng hallucination.
- `Audience Match Rate`: tỷ lệ top-5 có `audience` đúng kỳ vọng hoặc bằng `all`.

## Hạn chế hiện tại

Corpus hiện chỉ có `audience: student`. Vì vậy Audience Match Rate 100% chưa chứng minh hệ thống có thể tránh lấy nhầm tài liệu `faculty` hoặc `staff`. Trước khi kết luận về metric này, cần bổ sung tài liệu khác audience có provenance rõ ràng hoặc một bộ distractor được kiểm soát.
