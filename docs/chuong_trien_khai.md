# Chương: Triển khai ứng dụng và đánh giá

Nội dung dưới đây mô tả phiên bản hiện hành trong repository. Trước khi nộp, nhóm cần đối chiếu tên thành viên, thông tin học phần và yêu cầu trình bày của giảng viên.

## 1. Mục tiêu triển khai

Ứng dụng StrokeCare AI minh họa cách đưa pipeline học máy vào một giao diện nhập thông tin cá nhân và hiển thị điểm đầu ra của mô hình. Ứng dụng phục vụ trình diễn trong đồ án; không dùng để chẩn đoán, sàng lọc lâm sàng hoặc thay thế đánh giá của nhân viên y tế.

## 2. Kiến trúc xử lý

Ứng dụng được viết bằng Streamlit. Người dùng nhập mười trường tương ứng với các đặc trưng trong bộ dữ liệu: giới tính, tuổi, tăng huyết áp, bệnh tim, tình trạng hôn nhân, loại công việc, nơi cư trú, mức glucose trung bình, BMI và tình trạng hút thuốc. Nhãn hiển thị bằng tiếng Việt được ánh xạ về các giá trị gốc mà preprocessing đã học.

Các giá trị nhập được gom thành một dòng `DataFrame` với tên cột trùng dữ liệu huấn luyện. Nếu người dùng chọn không biết BMI, ứng dụng gửi `NaN`; pipeline xử lý thiếu trong bước imputation. Ứng dụng tải toàn bộ pipeline từ `models/stroke_pipeline.pkl`, nhờ đó bước biến đổi dữ liệu khi dự đoán được áp dụng nhất quán với lúc huấn luyện.

Pipeline gồm `ColumnTransformer`, bước SMOTE trong giai đoạn fit và Logistic Regression. Preprocessing dùng median cho đặc trưng số, thêm cờ thiếu cho BMI, chuẩn hóa biến số, one-hot encoding biến phân loại và bỏ qua nhãn phân loại chưa gặp. SMOTE chỉ được fit trong các fold huấn luyện; không áp dụng vào holdout.

## 3. Các bước dự đoán

1. Tải pipeline đã lưu bằng `joblib` khi ứng dụng khởi động.
2. Nhận mười trường từ form và chuyển nhãn hiển thị sang giá trị gốc.
3. Đóng gói dữ liệu thành một dòng có đúng tên cột.
4. Gọi `predict_proba` và hiển thị điểm lớp đột quỵ.
5. Hiển thị giải thích giới hạn: đầu ra chưa được hiệu chuẩn theo tỷ lệ thực tế và không phải xác suất y khoa.

Ngưỡng quyết định được đọc từ `results/final_model_metadata.json`. Với mô hình hiện tại, ngưỡng 0,74 được chọn trước trên dự đoán out-of-fold của training set bằng tiêu chí F1; nếu F1 hòa, ưu tiên Recall. Không chọn ngưỡng bằng cách tối ưu trên holdout.

## 4. Đánh giá mô hình

Vì nhãn đột quỵ chiếm tỷ lệ nhỏ, accuracy không đủ để mô tả chất lượng. Mã trong `src/evaluation.py` tính Recall, Precision, F1, ROC-AUC và Average Precision; đồng thời hỗ trợ confusion matrix, ROC curve, Precision-Recall curve và bảng Recall/Precision/F1 theo threshold.

Cross-validation dùng `StratifiedKFold` năm fold. Pipeline đặt preprocessing và SMOTE bên trong từng fold nhằm tránh rò rỉ dữ liệu. Cấu hình được chọn dựa trên Average Precision và các tiêu chí đã thống nhất; threshold được khảo sát bằng dự đoán out-of-fold. Sau khi khóa mô hình, holdout được đánh giá một lần và giữ nguyên kết quả.

Trong lần chạy lại trên preprocessing hiện tại, Logistic Regression đạt AP CV 0,226; XGBoost đã tuning đạt AP CV 0,214. Nhóm chọn Logistic Regression theo AP CV. Holdout đạt Recall 0,640, Precision 0,219, F1 0,327, ROC-AUC 0,845 và AP 0,275 tại threshold 0,74. Confusion matrix có TN=858, FP=114, FN=18, TP=32. Các biểu đồ và bảng chi tiết nằm trong `results/`.

## 5. Hạn chế và hướng sử dụng

- Bộ dữ liệu là dữ liệu học tập, không đại diện cho mọi quần thể hoặc cơ sở y tế.
- SMOTE thay đổi phân bố lớp trong quá trình huấn luyện; điểm `predict_proba` chưa được hiệu chuẩn có thể không tương ứng với xác suất thực tế.
- Chênh lệch AP CV hiện tại nhỏ; chưa có nested cross-validation hoặc kiểm định thống kê để xác nhận khác biệt.
- Kết quả chưa được xác thực độc lập ngoài mẫu dữ liệu này.
- Đầu ra không được dùng để tự chẩn đoán, thay đổi thuốc hoặc trì hoãn cấp cứu. Khi có dấu hiệu đột quỵ cấp tính, cần tìm hỗ trợ y tế khẩn cấp.

## 6. Cấu hình chạy

Môi trường và lệnh cài đặt được ghi trong `requirements.txt` và `HUONG_DAN_NHOM.md`. Pipeline cuối đã được lưu ở `models/stroke_pipeline.pkl`. Khởi chạy từ thư mục gốc bằng:

```powershell
streamlit run app.py
```

Ứng dụng có thể mở form khi thiếu pipeline, nhưng sẽ thông báo rằng chưa thể dự đoán. Pipeline hiện hành được fit trên toàn bộ training set với Logistic Regression và SMOTE trong pipeline; pipeline tạm trước đó không dùng làm kết quả báo cáo.
