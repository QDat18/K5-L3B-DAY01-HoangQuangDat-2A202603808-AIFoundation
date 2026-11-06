# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603808
- **Họ và tên:** Hoàng Quang Đạt
- **SEED** (in ở ô đầu notebook): 9225

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương án | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
|---|---|---|---|---|
| 1. BoW + LogReg (tự code) | 0.869 | 0.889 | 0.097 | 0 |
| 2. BoW + MLP | 0.927 | 0.947 | 0.156 | 0 |
| 3a. MiniLM + kNN | 0.917 | 0.947 | 53.171 | 0 |
| 3b. MiniLM + LogReg | 0.930 | 0.919 | 48.650 | 0 |
| 3c. MiniLM + MLP head | 0.942 | 0.919 | 36.873 | 0 |
| 4a. LLM nhỏ zero-shot | – | 0.950 | 4865.300 | 0.025 |
| 4c. LLM nhỏ few-shot | – | 0.974 | 5238.380 | 0.070 |

Đường Pareto (trên SUB, p95): BoW + LogReg, BoW + MLP, LLM zero-shot, LLM few-shot. Các cách đạt SLA p95 < 50 ms: hai mô hình BoW, 3b và 3c. Lưu ý SUB chỉ có 40 tin nên chênh dưới ~0.05 F1 coi như hòa.

## 3. Dự đoán và thực tế

1. **`emb_tricky` (Thí nghiệm 3.B):** đoán "3b đoán đúng nhiều hơn BoW"; thực tế hòa 5/6 vs 5/6, cùng sai tin "Last chance! Your account rewards expire tonight...". Lệch vì tin này không chứa từ khoá spam mạnh (BoW) và giọng văn giống thông báo hợp lệ (embedding); ngoài ra 6 tin quá ít để phân biệt hai mô hình.
2. **`tradeoff`:** đoán "BoW + LogReg và cả hai LLM nằm bên phải vạch SLA"; thực tế BoW + LogReg có p95 chỉ 0.097 ms, nằm rất xa bên trái vạch (nhanh nhất), chỉ hai LLM (p95 ~5 giây) vượt SLA. Lệch vì mình nhầm mô hình BoW là chậm, trong khi nó chỉ là phép cộng trọng số.
3. **`dl_curve` / `hard_eval`:** ở `hard_eval` mình đoán LLM yếu nhất ở nhóm prompt injection, nhưng thực tế cả 4a và 4d đạt 100% ở nhóm này (và 98%/100% tổng). Lệch vì prompt vốn đã yêu cầu trả đúng một nhãn và LLM nhỏ vẫn nhận ra ý đồ lừa đảo khi đọc cả câu.

## 4. Phân tích 50 case khó

Tụt nhiều nhất: **BoW + LogReg** (F1 từ 0.889 xuống 0.333, accuracy 52%) và **BoW + MLP** (0.947 xuống 0.400, accuracy 52%). Nhóm khó nhất với BoW là Né từ khoá (LogReg 17%, MLP 0%), Spam giọng tự nhiên (10%) và Thao túng/lừa đảo (12–38%). Embedding khó nhất ở Spam giọng tự nhiên (10–30%) và Nói về spam (25–50%); LLM gần như không sai (98–100%).

Hai case cụ thể:
- **#29** `C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd...` (né từ khoá): BoW tách thành các token như "c0ngr4tulat10ns" không có trong từ điển 2000 từ nên bị bỏ qua, mất toàn bộ tín hiệu spam. LLM đọc như người nên hiểu được ý nghĩa.
- **#11** `Hey it's Jess from the gym 😊 ... 3 months for the price of 1, just reply JOIN` (spam giọng tự nhiên): không có từ khoá spam mạnh, nhiều từ giống tin thường ("hey", "it's", "from"), nên BoW và embedding đều nghiêng về ham; chỉ LLM nhận ra ý đồ quảng cáo.

Test thường lạc quan vì cùng phân phối với train (spam lộ liễu, nhiều từ khoá quen), nên mô hình thuộc từ khoá là đủ điểm. 50 case viết tay không đại diện spam ngoài đời; chỉ nên dùng để kiểm tra điểm yếu theo nhóm, không dùng làm con số F1 "thực tế".

## 5. Thử thách cá nhân

Đề #4 (SEED 9225 % 6 = 3): chạy `knn_predict` với k = 1, 5, 25 trên embedding MiniLM.

| k | precision | recall | F1 |
|---|---|---|---|
| 1 | 0.888 | 0.926 | 0.906 |
| 5 | 0.898 | 0.936 | 0.917 |
| 25 | 0.870 | 0.926 | 0.897 |

Nhận xét: k = 5 tốt nhất. k = 1 nhạy với nhiễu nên precision thấp hơn; k = 25 lấy phiếu cả từ láng giềng ở xa nên kéo thêm tin ham ở rìa vùng spam, precision giảm còn 0.870 (recall gần như không đổi, khác với dự đoán ban đầu của mình là recall sẽ tụt). Chênh lệch chỉ khoảng 2 điểm F1 trên 775 tin test (94 spam), nên chỉ kết luận chắc là k vừa phải tốt hơn hai thái cực.

## 6. Phương án cho sản phẩm thật

Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày, mình chọn **BoW + MLP**: F1 test 0.927, p95 ≈ 0.16 ms, chi phí 0 USD, nằm trên đường Pareto. 3c (F1 0.942, p95 36.9 ms) chỉ hơn khoảng 0.015 F1 (trong vùng hòa) nhưng chậm hơn khoảng 230 lần. LLM không đạt SLA (p95 ~5 giây) và tốn khoảng 0.025–0.070 USD/1k tin, tức khoảng 250–700 USD/ngày cho 10 triệu tin. Đánh đổi chấp nhận: BoW dễ bị qua mặt bởi spam né từ khoá và giọng tự nhiên (stress test chỉ đúng 52%). Để giảm rủi ro, đề xuất kiến trúc lai: BoW + MLP chạy trước; chỉ các tin có xác suất nằm vùng lưỡng lự (ví dụ 0.15–0.85) mới gửi sang LLM, kèm chuẩn hoá ký tự (leetspeak, unicode) trước BoW và lọc cụm "ignore previous instructions" trước LLM.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời gì | Mình sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 4 — `llm_classify_batch` | Viết hàm phân loại cả loạt, đo độ trễ và chi phí | Vòng lặp gọi `retry_with_backoff(call_llm)`, ước token = số từ / 0.75 | Chạy bộ kiểm tra 4.4 trên mock và đọc lại công thức giá |
| Phần 4 — `parse_label` | Bắt lỗi hàm `parse_label_v2` | Chỉ ra lỗi "not spam" bị tính là spam, không trả `None`, crash với `None` | Thêm kiểm tra phủ định trước, chạy test 4.2–4.3 |
| Phần 5 — Pareto | Dự đoán và đọc biểu đồ Pareto | Giải thích đường Pareto, chỉ ra mô hình bị lấn át | Đối chiếu với biểu đồ thật, sửa lại dự đoán `tradeoff` của mình |
| Thử thách | Nhận xét kết quả kNN k = 1, 5, 25 | Nhận xét kết quả và chỉ ra số F1 trong notebook chưa khớp | Thay bằng số thật 0.906 / 0.917 / 0.897 |
| Tính tay | Cách tính 5 phép tính | Công thức từng bước | Tự tính lại trên giấy để chụp ảnh |

Điều hiểu ra hôm nay: (1) điểm cao trên test cùng phân phối không chứng minh mô hình "hiểu" spam, stress test cho thấy BoW tụt từ 0.87 xuống 0.33; (2) chọn mô hình là bài toán đánh đổi chất lượng - độ trễ - chi phí, không phải chọn mô hình có F1 cao nhất, và những chênh lệch nhỏ trên tập 40 tin không đáng tin; (3) dự đoán sai (như `tradeoff` hay `hard_eval`) là cơ hội để sửa trực giác, nên mình ghi lại thay vì chỉnh cho đẹp.