# Dàn ý thuyết trình đồ án Stroke Prediction

## 1. Bài toán và mục tiêu

Nêu đầu vào, nhãn đột quỵ và mục tiêu so sánh các mô hình phân lớp trong phạm vi học thuật.

## 2. Dữ liệu

5.110 bản ghi; 249 ca có nhãn stroke (4,87%); BMI thiếu 201 giá trị. Dùng một biểu đồ phân phối nhãn và nêu rủi ro mất cân bằng.

## 3. EDA

Chọn biểu đồ tuổi, glucose, BMI và một biến phân loại có thông điệp rõ. Trình bày quan sát, tránh diễn giải quan hệ nhân quả.

## 4. Tiền xử lý

Sơ đồ luồng: chia train/test có stratify, fit imputer/encoder/scaler trong fold, áp dụng SMOTE riêng ở phần train của mỗi fold.

## 5. Sáu thuật toán

Giới thiệu ngắn các nhóm tuyến tính, lân cận/biên và cây. Nêu vì sao AP/Recall phù hợp hơn accuracy đơn lẻ.

## 6. Kết quả cross-validation

Trình bày bảng sáu mô hình từ `results/cv_results.csv`. Trong lần chạy hiện tại, Logistic Regression có AP 0,226; XGBoost cơ bản có AP 0,151.

## 7. Tuning và so sánh mô hình

Tuning đạt AP 0,177 với Random Forest và 0,214 với XGBoost. So sánh cách giải thích, khả năng học phi tuyến và chi phí tuning trong `docs/so_sanh_mo_hinh.md`. Nhóm chọn Logistic Regression theo AP CV.

## 8. Đánh giá cuối

Threshold 0,74 được chọn theo F1 trên dự đoán OOF. Holdout một lần: Recall 0,640, Precision 0,219, F1 0,327, ROC-AUC 0,845, AP 0,275; confusion matrix TN=858, FP=114, FN=18, TP=32. Dùng các biểu đồ trong `results/`; không đưa điểm test cũ dùng preprocessing trước khi sửa vào kết luận.

## 9. Ứng dụng

Chụp màn hình form StrokeCare AI, mô tả cách pipeline nhận dữ liệu và điểm đầu ra. Nhắc đây là minh họa học thuật, không phải chẩn đoán y tế.

## 10. Hạn chế và kết luận

Nêu dữ liệu mất cân bằng, xác suất sau SMOTE chưa hiệu chuẩn và chưa có kiểm định ngoài. Kết luận dựa trên metric hiện hành, không khẳng định khả năng sử dụng lâm sàng.
