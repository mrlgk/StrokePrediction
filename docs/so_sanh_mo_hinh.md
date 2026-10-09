# So sánh Logistic Regression và XGBoost

## Kết quả cùng phiên bản preprocessing

Các số dưới đây được tính trên training set bằng Stratified 5-Fold CV. Logistic Regression là cấu hình mặc định trong danh sách sáu mô hình; XGBoost được tuning bằng RandomizedSearchCV theo Average Precision.

| Mô hình | Recall CV | Precision CV | F1 CV | ROC-AUC CV | Average Precision CV |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.779 | 0.140 | 0.237 | 0.848 | 0.226 |
| XGBoost cơ bản | 0.110 | 0.187 | 0.138 | 0.801 | 0.151 |
| XGBoost sau tuning | — | — | — | — | 0.214 |

Các số cũ trong nhánh modeling là Logistic Regression AP 0.192 và XGBoost tuning AP 0.1606. Chạy lại trên preprocessing đã tích hợp vào nhánh app làm thay đổi kết quả thành 0.226 và 0.214. Dùng kết quả chạy lại khi viết báo cáo cuối.

Ở lần chạy cũ, Logistic Regression cao hơn 0.0314 điểm AP (xấp xỉ 19,6% so với 0.1606). Nếu hai con số cũ được lấy từ cùng quy trình CV, đây là lợi thế quan sát được của Logistic Regression theo AP; tuy vậy, điểm `best_score_` của tìm kiếm siêu tham số có thể lạc quan hơn vì nó là điểm cao nhất trong quá trình chọn cấu hình. Tỷ lệ lớp dương toàn bộ dữ liệu là 4,87%, làm mốc tham khảo cho AP của bộ dữ liệu mất cân bằng.

Kết quả holdout lịch sử của XGBoost, được tạo bằng preprocessing cũ, được lưu riêng ở `results/legacy_test_results_before_preprocessing_fix.csv` và không dùng trong kết luận. Kết quả holdout của pipeline hiện tại nằm ở `results/test_results.csv`.

## Logistic Regression

Mô hình tính xác suất qua hàm sigmoid trên tổ hợp tuyến tính của các đặc trưng. Ưu điểm là huấn luyện nhanh, ít siêu tham số và giải thích được chiều/hệ số của đặc trưng trong không gian đã mã hóa. Đây là lựa chọn phù hợp làm baseline và, theo AP CV hiện tại, cũng là cấu hình tốt nhất trong các kết quả đã chạy.

Hạn chế là biên quyết định cơ bản tuyến tính, nên không tự biểu diễn tốt các quan hệ phi tuyến hoặc tương tác phức tạp. Precision CV 0.140 còn thấp: khi chọn ngưỡng theo mặc định, nhiều ca dự đoán dương có thể là false positive. Hệ số cũng không chứng minh quan hệ nhân quả; SMOTE có thể làm xác suất đầu ra lệch khỏi tỷ lệ thực tế.

## XGBoost

XGBoost cộng dồn các cây theo boosting; các cây sau tập trung giảm lỗi của tổ hợp trước. Mô hình có thể biểu diễn quan hệ phi tuyến và tương tác, hỗ trợ regularization, và cung cấp feature importance. Tuning đã nâng AP CV của XGBoost từ 0.151 lên 0.214.

Hạn chế là nhiều siêu tham số hơn, thời gian tuning lớn hơn và khó giải thích trực tiếp hơn Logistic Regression. Tuning cũng có thể overfit vào các fold được dùng để tìm cấu hình; `best_score_` là kết quả tốt nhất trong tìm kiếm và không hoàn toàn ngang hàng với điểm CV của một mô hình cấu hình sẵn. SMOTE cũng khiến điểm `predict_proba` cần được đọc thận trọng.

## Kết quả cuối trên holdout

Nhóm đã chọn Logistic Regression theo AP CV. Threshold 0,74 được xác định trước khi đánh giá test, bằng cách tối đa hóa F1 trên dự đoán OOF của train; nếu F1 hòa, ưu tiên Recall. Trên holdout, Recall = 0,640, Precision = 0,219, F1 = 0,327, ROC-AUC = 0,845 và AP = 0,275. Confusion matrix gồm TN=858, FP=114, FN=18, TP=32.

## Cách diễn giải và quyết định

Theo tiêu chí Average Precision trên training CV, Logistic Regression dẫn 0,012 so với XGBoost sau tuning (0,226 so với 0,214). Độ lệch chuẩn giữa các fold của Logistic Regression là 0,041; nhóm chưa chạy kiểm định chênh lệch hoặc nested CV, vì vậy chưa thể kết luận chênh lệch nhỏ này có ý nghĩa thống kê. Nhóm chọn Logistic Regression theo tiêu chí AP CV. Kết quả holdout được báo riêng và không dùng để đổi mô hình hoặc ngưỡng.

AP là diện tích dưới đường Precision–Recall và không gắn với một threshold cố định. Threshold 0,74 là lựa chọn theo F1 của OOF trên train, không phải ngưỡng lâm sàng. Holdout chỉ đánh giá hiệu năng cho cấu hình đã khóa. Vì mô hình dùng SMOTE và chưa được hiệu chuẩn, điểm `predict_proba` không nên diễn giải là xác suất đột quỵ thực tế.
