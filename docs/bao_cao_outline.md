# Khung báo cáo đồ án Stroke Prediction

Khung này bám theo bố cục báo cáo mẫu bạn gửi. Các trường trong ngoặc vuông cần nhóm điền từ thông tin chính thức; nội dung định lượng phải khớp notebook và CSV trong repo.

## Trang bìa

- `[Tên trường và khoa]`
- `Báo cáo đồ án học máy: Dự đoán nguy cơ đột quỵ`
- `[Tên học phần]`, `[giảng viên hướng dẫn]`, `[thành viên và mã số sinh viên]`
- `[lớp]`, `[địa điểm]`, `[năm học]`

## Phần đầu

1. Bảng phân công nhiệm vụ.
2. Lời cảm ơn ngắn.
3. Mục lục tự động.
4. Danh mục hình, bảng và thuật ngữ nếu cần.

## I. Giới thiệu

Nêu bài toán phân loại nhị phân trên bộ dữ liệu đột quỵ, mục tiêu học thuật, phạm vi thử nghiệm và giới hạn sử dụng. Tránh mô tả ứng dụng như công cụ chẩn đoán.

## II. Phân tích dữ liệu và tiền xử lý

- Nguồn, số dòng/cột, ý nghĩa nhãn và phạm vi dữ liệu.
- Chất lượng dữ liệu: thiếu BMI, mất cân bằng nhãn, giá trị phân loại `Unknown`, kiểm tra bản ghi trùng và cột mã `id`.
- Năm nhóm biểu đồ EDA, kèm nhận xét mô tả và không suy diễn nhân quả.
- Chia train/test có stratify trước khi fit preprocessing.
- Imputation, cờ thiếu, chuẩn hóa biến số, mã hóa biến phân loại và xử lý mất cân bằng trong pipeline.

## III. Cơ sở lý thuyết và thiết kế

Tóm tắt Logistic Regression, Decision Tree, KNN, SVM, Random Forest và XGBoost theo `docs/chuong_ly_thuyet.md`. Mô tả pipeline, Stratified 5-Fold CV, RandomizedSearchCV, metric cho lớp hiếm và cách chọn threshold bằng out-of-fold trên train.

## IV. Cài đặt ứng dụng

Trình bày kiến trúc trong `docs/chuong_trien_khai.md`: cấu trúc đầu vào, ánh xạ nhãn, tải pipeline, luồng dự đoán và giao diện Streamlit. Chèn ảnh chụp ứng dụng sau khi nhóm chạy bằng pipeline cuối.

## V. Thử nghiệm và đánh giá

- Bảng CV sáu mô hình với trung bình và độ lệch chuẩn.
- Kết quả tuning Random Forest và XGBoost.
- Mô hình cuối và threshold đã chốt trước khi mở holdout.
- Bảng metric test, confusion matrix, ROC và Precision-Recall.
- Feature importance/permutation importance của mô hình cuối.
- Kết quả kiểm thử ca biên. Phân biệt kiểm tra kỹ thuật pipeline tạm với kết quả mô hình cuối.

## Kết luận và hướng phát triển

Tóm tắt kết quả đúng theo CSV hiện hành. Nêu giới hạn về dữ liệu học tập, mất cân bằng, xác suất chưa hiệu chuẩn và thiếu đánh giá ngoài. Chỉ đề xuất kiểm định thêm như hướng tiếp theo.

## Tài liệu tham khảo và phụ lục

Ghi nguồn bộ dữ liệu, thư viện và tài liệu thuật toán đã thực sự dùng. Đưa hướng dẫn chạy, bảng ca biên và bảng kết quả đầy đủ vào phụ lục khi cần.

## Còn cần điền trước khi phát hành

- Thông tin bìa và thành viên nhóm.
- Ảnh chụp ứng dụng thực tế nếu cần theo mẫu của giảng viên.
- Nguồn tài liệu tham khảo theo chuẩn trích dẫn giảng viên yêu cầu.
