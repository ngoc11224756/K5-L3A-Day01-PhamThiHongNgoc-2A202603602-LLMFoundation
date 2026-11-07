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
