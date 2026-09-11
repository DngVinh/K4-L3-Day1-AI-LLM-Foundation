# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature 0.0, các phản hồi thường nhất quán, tập trung và có xu hướng lặp lại cùng một ý chính. Khi tăng temperature lên 0.5, 1.0 rồi 1.5, cách diễn đạt và lựa chọn ý tưởng thường đa dạng, sáng tạo hơn nhưng cũng dễ lan man hoặc kém ổn định hơn. Tham số này điều chỉnh mức độ ngẫu nhiên khi model chọn token, chứ không trực tiếp làm tăng kiến thức hay năng lực suy luận của model.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ bắt đầu với temperature khoảng 0.2–0.3 cho chatbot hỗ trợ khách hàng. Mức thấp giúp câu trả lời nhất quán, dễ kiểm soát và hạn chế việc model tự thêm thông tin chưa được xác minh, đồng thời vẫn đủ tự nhiên cho giao tiếp lịch sự. Với các tác vụ thiên về sáng tạo như viết nội dung quảng bá, tôi sẽ đánh giá riêng và có thể tăng temperature sau khi kiểm thử chất lượng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Mỗi ngày có 10.000 × 3 = 30.000 lượt gọi, tương đương 30.000 × 350 = 10.500.000 token đầu ra. Theo bảng giá của bài, GPT-4o có chi phí khoảng 10.500 × 0,010 = 105 USD, còn GPT-4o-mini khoảng 10.500 × 0,0006 = 6,30 USD; như vậy GPT-4o đắt hơn khoảng 16,7 lần, chưa tính token đầu vào. GPT-4o phù hợp với các tác vụ phức tạp, đòi hỏi suy luận hoặc độ chính xác cao, trong khi GPT-4o-mini phù hợp hơn cho FAQ, phân loại và các câu trả lời ngắn nhằm tối ưu chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, câu trả lời thường ngắn hơn, dùng từ vựng đơn giản và có thể minh họa blockchain bằng hình ảnh một quyển sổ được nhiều người cùng lưu giữ. Với persona chuyên gia tài chính, câu trả lời có xu hướng chuyên sâu hơn, sử dụng các thuật ngữ như sổ cái phân tán, cơ chế đồng thuận và tính bất biến, đồng thời phân tích thêm ưu điểm hoặc rủi ro. Như vậy, system prompt định hướng vai trò, giọng điệu, độ sâu và cách trình bày của model; tuy nhiên, nó không tự bổ sung kiến thức hoặc bảo đảm thông tin luôn chính xác.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với một đoạn tiếng Việt khoảng 100 từ, công thức ước lượng của Part 1 cho 100 / 0,75 ≈ 133 token. Trong một phép đo minh họa, nếu `count_tokens` trả về khoảng 160 token thì chênh lệch là (160 − 133) / 133 ≈ 20% so với cách ước lượng thô. Đây chỉ là ví dụ, vì kết quả phụ thuộc vào nội dung và tokenizer; tiếng Việt thường có thể tốn nhiều token hơn do dấu thanh, ký tự có dấu và cách phân tách âm tiết không luôn trùng với các token phổ biến trong dữ liệu huấn luyện.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming đặc biệt hữu ích trong các tương tác trực tiếp có câu trả lời dài, chẳng hạn chatbot, trợ lý viết nội dung hoặc công cụ phân tích tài liệu, vì người dùng nhìn thấy token đầu tiên sớm và có thể theo dõi tiến độ. Non-streaming phù hợp hơn với câu trả lời ngắn, tác vụ chạy nền hoặc quy trình cần nhận đủ một JSON hoàn chỉnh trước khi xử lý bước tiếp theo. Streaming chủ yếu cải thiện thời gian chờ có cảm nhận; nó không nhất thiết làm giảm tổng thời gian model sinh toàn bộ câu trả lời.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff khiến thời gian chờ giữa các lần retry tăng dần, giúp server có thêm thời gian hồi phục và phân tán lưu lượng thay vì tiếp tục nhận request dồn dập. Nếu hàng nghìn client cùng chờ đúng 1 giây rồi retry, chúng sẽ tạo ra hiện tượng “thundering herd”: server vừa hồi phục đã bị dồn thêm một đợt request, khiến tình trạng quá tải có thể kéo dài. Trong hệ thống thực tế, nên kết hợp thêm jitter ngẫu nhiên để giảm khả năng các client retry đồng bộ.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona tôi chọn: **“Bạn là trợ lý học lập trình thân thiện. Hãy trả lời bằng tiếng Việt, giải thích từng bước bằng ví dụ ngắn, ưu tiên câu trả lời dài 3–5 câu và nói rõ khi không chắc chắn thay vì tự bịa thông tin.”** Yêu cầu “bằng tiếng Việt” bảo đảm đầu ra phù hợp với đối tượng sử dụng, còn giới hạn “3–5 câu” giúp kiểm soát độ dài, chi phí và khả năng đọc hiểu. Quy tắc thừa nhận sự không chắc chắn cũng giúp hạn chế những câu trả lời có vẻ tự tin nhưng thiếu căn cứ.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất hiện nay là history chỉ giữ ba lượt gần nhất, vì vậy trợ lý có thể quên các quyết định hoặc dữ kiện quan trọng ở đầu cuộc trò chuyện. Tôi sẽ bổ sung một `conversation_summary`: trước khi loại bỏ message cũ, hệ thống tóm tắt mục tiêu, dữ kiện và quyết định chính, rồi đưa bản tóm tắt vào context cùng sáu message gần nhất. Nếu cần hỗ trợ nhiều phiên, bản tóm tắt có thể được lưu theo `conversation_id` và nạp lại khi người dùng quay trở lại.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
