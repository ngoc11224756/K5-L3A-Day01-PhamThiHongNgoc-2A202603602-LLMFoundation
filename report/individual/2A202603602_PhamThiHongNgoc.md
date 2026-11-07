# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202603602**
- **Họ và tên: Phạm Thị Hồng Ngọc**
- **SEED** (in ở ô đầu notebook): 9952

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅|

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
|---|---|---|---|---|
| 1. BoW + LogReg (tự code) | 0.877 | 0.857 | 0.100 | 0.0 |
| 2. BoW + MLP (deep learning) | 0.945 | 0.889 | 0.352 | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 0.906 | 0.923 | 62.693 | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 0.959 | 0.974 | 64.724 | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 0.959 | 0.974 | 65.109 | 0.0 |

## 3. Dự đoán và thực tế
3 dự đoán tôi chọn:
1. dl_curve – So sánh MLP và Logistic Regression

Dự đoán: Ban đầu, em nghĩ MLP sẽ có kết quả F1 gần tương đương Logistic Regression vì cả hai đều được huấn luyện trên cùng bộ dữ liệu.

Kết quả thực tế: MLP đạt F1 = 0.945, cao hơn Logistic Regression (0.877). Recall của MLP cũng đạt 0.897, so với 0.782 của Logistic Regression.

Vì sao lệch: Em chưa đánh giá đúng khả năng của MLP. MLP có thể học được những mối quan hệ phức tạp giữa các từ nên nhận diện được nhiều tin nhắn spam hơn Logistic Regression.

2. emb_tricky – So sánh embedding và BoW

Dự đoán: Em nghĩ mô hình sử dụng embedding (3b) sẽ phân loại chính xác hơn BoW vì embedding có khả năng nắm bắt ý nghĩa ngữ cảnh của từ.

Kết quả thực tế: Hai phương pháp cho kết quả ngang nhau, cùng phân loại đúng 4/6 tin nhắn.

Vì sao lệch: Em nghĩ embedding sẽ tốt hơn, nhưng thực tế mỗi phương pháp nhận diện tốt những cụm từ khác nhau. Hơn nữa, chỉ thử trên 6 tin nhắn nên chưa thể kết luận embedding tốt hơn BoW.

3. ml_lr – Ảnh hưởng của learning rate

Dự đoán: Em dự đoán learning rate = 50 sẽ khiến loss dao động mạnh trong quá trình huấn luyện do bước cập nhật trọng số quá lớn.

Kết quả thực tế: Loss chỉ tăng vọt lên khoảng 2.7 trong vài epoch đầu, sau đó giảm nhanh về khoảng 0.004.

Vì sao lệch: Em nghĩ learning rate lớn sẽ khiến loss liên tục dao động, nhưng thực tế mô hình chỉ bị tăng loss lúc đầu rồi nhanh chóng giảm xuống. Có thể do dữ liệu trong thí nghiệm khá dễ phân loại nên mô hình nhanh chóng học được cách phân biệt hai nhóm.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
## AI Usage Log

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| TODO 1.1 – `tokenize`, `vectorize` | Hỏi cách triển khai `vectorize`, cách sử dụng `tokenize(texts[i])` và ý nghĩa các bước xử lý | Giải thích logic từng bước, gợi ý cách viết code nhưng không đưa lời giải hoàn chỉnh | Tự viết code, sửa lỗi `NoneType` do thiếu `return`, loại bỏ `raise` và `print` không cần thiết |
| TODO 1.2 – `sigmoid`, `predict_proba` | Hỏi về công thức sigmoid, cách chuyển giá trị đầu ra thành xác suất và cách hoàn thiện hàm | Giải thích công thức sigmoid, chỉ ra lỗi `return 1` và các phần logic chưa đúng | Tự sửa hàm `sigmoid`, chạy thử và kiểm tra kết quả |
| TODO 1.3 – `train_logreg` | Hỏi cách triển khai hàm huấn luyện Logistic Regression và ý nghĩa các bước gradient descent | Phân tích quy trình 4 bước, giải thích cập nhật trọng số `w`, bias `b` và lưu `losses` | Tự viết các dòng code còn thiếu và kiểm tra quá trình huấn luyện |
| TODO 1.4 – `metrics` | Hỏi cách tính TP, FP, FN, TN, Precision, Recall và F1-score | Giải thích ý nghĩa từng chỉ số, công thức tính và nguyên nhân lỗi `ZeroDivisionError` | Tự viết logic đếm và công thức tính các chỉ số |
| TODO 2.1, 2.2 – `build_mlp`, `train_mlp` | Hỏi về cấu trúc MLP, cách xây dựng mô hình và quy trình huấn luyện | Giải thích cấu trúc hàm, vai trò các thành phần và gợi ý 5 bước triển khai | Tự điền các phần code còn thiếu, chạy và kiểm tra kết quả |
| Embedding | Hỏi về cách biểu diễn văn bản thành vector, ý nghĩa của embedding và các phép tính liên quan | Giải thích khái niệm, cơ chế biểu diễn và ý nghĩa của các vector | Tự thực hiện code và đối chiếu kết quả |
| Công thức ML/DL | Hỏi cách hiểu và áp dụng các công thức, ý nghĩa biến số và trình tự tính toán | Phân tích công thức, giải thích từng thành phần và liên hệ với code trong notebook | Tự đọc hiểu, áp dụng và kiểm tra kết quả tính toán |
| Phân tích kết quả | Hỏi cách đọc kết quả thực nghiệm, so sánh mô hình và giải thích các chỉ số đánh giá | Hướng dẫn cách diễn giải kết quả và gợi ý các yếu tố cần xem xét | Tự chạy thực nghiệm, ghi nhận số liệu và đưa ra nhận xét |
| Câu hỏi lý thuyết | Hỏi về data leakage, learning rate, overfitting và các vấn đề liên quan đến ML/DL | Giải thích bản chất, nguyên nhân, hậu quả và gợi ý các ý chính để trả lời | Tự tổng hợp và diễn đạt lại bằng lời của mình |
| Gỡ lỗi | Gửi thông báo lỗi hoặc đoạn code chưa chạy đúng để hỏi nguyên nhân và cách khắc phục | Phân tích lỗi, giải thích nguyên nhân và hướng dẫn các bước kiểm tra | Tự chỉnh sửa code, chạy lại và xác nhận kết quả |
| Hoàn thiện bài nộp | Hỏi cách kiểm tra TODO, chuẩn bị notebook, file báo cáo và cấu trúc thư mục nộp bài | Hướng dẫn kiểm tra các yêu cầu của BTC và quy trình upload lên GitHub | Tự kiểm tra, chuẩn bị file và commit lên repository |

Qua bài lab này, em được ôn lại cách làm việc với github 1 cách rõ ràng và chuyên nghiệp hơn. Hiểu thêm về các mô hình học máy
