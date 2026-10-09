# Cơ sở lý thuyết các thuật toán phân lớp

## 1. Logistic Regression

Logistic Regression ước lượng xác suất một mẫu thuộc lớp dương. Với vector đặc trưng \(x\), mô hình tính tổ hợp tuyến tính \(z = w^T x + b\), sau đó đưa giá trị này qua hàm sigmoid \(p(y=1|x)=1/(1+e^{-z})\). Mô hình học các hệ số bằng cách tối thiểu hóa log-loss; regularization có thể hạn chế hệ số quá lớn và giảm overfitting.

Ưu điểm của Logistic Regression là tốc độ huấn luyện nhanh, kết quả xác suất dễ diễn giải và hệ số cho biết hướng liên hệ giữa từng đặc trưng với log-odds khi các điều kiện khác giữ nguyên. Mô hình tạo đường phân chia tuyến tính trong không gian đặc trưng, nên có thể bỏ sót quan hệ phi tuyến hoặc tương tác phức tạp nếu không bổ sung đặc trưng phù hợp. Việc chuẩn hóa biến số thường giúp tối ưu ổn định hơn.

Trong bài toán này, Logistic Regression là mốc so sánh gọn và dễ giải thích. Dữ liệu mất cân bằng được xử lý trong pipeline bằng SMOTE trên từng fold huấn luyện. Mô hình được đánh giá bằng Recall, Precision, F1, ROC-AUC và Average Precision; threshold mặc định 0,5 không được xem là lựa chọn bắt buộc.

## 2. Decision Tree

Decision Tree chia dữ liệu thành các nhóm bằng những quy tắc dạng “đặc trưng này nhỏ hơn ngưỡng”. Ở mỗi nút, thuật toán tìm phép chia làm các nhánh con thuần hơn theo tiêu chí như Gini impurity hoặc entropy. Quá trình tiếp tục cho đến khi đạt giới hạn độ sâu, số mẫu tối thiểu hoặc điều kiện dừng khác. Dự đoán ở lá thường là lớp phổ biến hoặc tỷ lệ lớp của các mẫu trong lá.

Cây quyết định có thể mô tả bằng các quy tắc dễ đọc, xử lý quan hệ phi tuyến và không đòi hỏi chuẩn hóa. Tuy vậy, cây sâu có thể ghi nhớ nhiễu; những thay đổi nhỏ trong dữ liệu cũng có thể tạo cấu trúc cây khác. Giới hạn độ sâu, số mẫu tối thiểu mỗi lá và pruning giúp kiểm soát độ phức tạp.

Với dữ liệu stroke có lớp dương hiếm, một cây chưa cân bằng có thể ưu tiên lớp đa số và bỏ sót nhiều ca dương. Vì vậy, cần đặt bước xử lý mất cân bằng trong pipeline và theo dõi Recall cùng Precision-Recall thay vì chỉ nhìn accuracy. Cây đơn lẻ được dùng làm mô hình dễ giải thích và mốc phi tuyến trong so sánh.

## 3. K-Nearest Neighbors

K-Nearest Neighbors (KNN) dự đoán nhãn của một điểm dựa trên nhãn của \(k\) điểm huấn luyện gần nó nhất. Khoảng cách Euclidean thường dùng cho dữ liệu số; các biến phân loại cần được mã hóa trước. Phân loại có thể lấy lớp chiếm đa số trong lân cận hoặc dùng trọng số lớn hơn cho điểm ở gần.

KNN có cách học đơn giản: mô hình gần như lưu lại dữ liệu huấn luyện thay vì ước lượng một bộ tham số lớn. Cách này có thể biểu diễn ranh giới phức tạp, nhưng dự đoán tốn thời gian khi dữ liệu lớn và nhạy với thang đo, nhiễu cùng giá trị \(k\). Chuẩn hóa biến số và chọn \(k\) phù hợp là các bước quan trọng.

KNN nhạy với mất cân bằng vì vùng lân cận có thể bị áp đảo bởi lớp đa số. Trong đồ án, preprocessing và SMOTE được đặt trong pipeline để tránh tạo mẫu tổng hợp trước khi chia fold. KNN được đánh giá theo cùng các fold và metric với những thuật toán còn lại để so sánh công bằng.

## 4. Support Vector Machine

Support Vector Machine (SVM) tìm siêu phẳng phân chia hai lớp sao cho biên giữa siêu phẳng và các điểm gần nhất được tối đa hóa. Với dữ liệu không thể chia tuyến tính, kernel như RBF ánh xạ quan hệ giữa điểm dữ liệu sang không gian đặc trưng khác. Tham số \(C\) điều chỉnh mức phạt cho lỗi phân loại; kernel RBF còn dùng \(\gamma\) để kiểm soát phạm vi ảnh hưởng của từng điểm.

SVM có thể hoạt động tốt trong không gian nhiều chiều và chỉ một phần điểm, gọi là support vectors, quyết định biên. Đổi lại, việc chọn kernel và tham số có ảnh hưởng lớn; mô hình khó diễn giải trực tiếp hơn cây hoặc hồi quy. Chuẩn hóa biến số đặc biệt quan trọng vì thuật toán dựa trên khoảng cách và tích vô hướng.

SVM được đưa vào so sánh với các mô hình tuyến tính, lân cận và cây. Khi cần xác suất, cấu hình SVC phải bật hiệu chuẩn xác suất hoặc sử dụng một quy trình hiệu chuẩn riêng; kết quả xác suất cần được kiểm tra trước khi dùng để phân tích threshold. Recall, F1 và Average Precision phản ánh tốt hơn mục tiêu phát hiện lớp stroke hiếm.

## 5. Random Forest

Random Forest kết hợp nhiều Decision Tree. Mỗi cây được huấn luyện trên một mẫu bootstrap và tại mỗi nút chỉ xem xét một tập con đặc trưng ngẫu nhiên. Phân loại cuối cùng lấy phiếu đa số hoặc trung bình xác suất từ các cây. Sự đa dạng giữa các cây giúp giảm phương sai so với một cây sâu đơn lẻ.

Random Forest thường nắm bắt quan hệ phi tuyến, tương tác và ít cần chuẩn hóa. Mô hình vẫn có thể thiên về lớp đa số nếu dữ liệu mất cân bằng; tăng số cây không tự giải quyết vấn đề này. Độ sâu, số mẫu mỗi lá, số đặc trưng xét tại mỗi nút và cơ chế cân bằng dữ liệu ảnh hưởng đến kết quả.

Trong đồ án, Random Forest được tuning bằng RandomizedSearchCV với Average Precision làm tiêu chí. Mỗi fold dùng pipeline gồm preprocessing, SMOTE và mô hình, để dữ liệu tổng hợp chỉ xuất hiện trong phần huấn luyện của fold. Feature importance dựa trên mức giảm impurity có thể làm nổi bật đặc trưng liên tục hoặc có nhiều mức; permutation importance được bổ sung để kiểm tra mức giảm điểm số khi xáo trộn một đặc trưng.

## 6. XGBoost

XGBoost là phương pháp boosting dựa trên cây. Các cây được thêm lần lượt; mỗi bước tối ưu hàm mục tiêu gồm loss và phần phạt độ phức tạp. Những cây sau tập trung giảm lỗi còn lại của tổ hợp cây trước. Learning rate, số cây, độ sâu, subsampling và regularization điều chỉnh khả năng học và nguy cơ overfitting.

Boosting có thể nắm bắt quan hệ phi tuyến và tương tác mà không cần chuẩn hóa như các phương pháp dựa trên khoảng cách. Hiệu năng phụ thuộc vào cấu hình siêu tham số và quy trình xác thực; nếu dùng quá nhiều cây hoặc cây quá sâu, mô hình có thể khớp nhiễu. XGBoost cũng không tự bảo đảm xác suất được hiệu chuẩn theo tỷ lệ stroke thực tế, đặc biệt khi kết hợp SMOTE.

Đồ án dùng RandomizedSearchCV để tìm cấu hình theo Average Precision. SMOTE nằm trong pipeline để không rò rỉ thông tin qua các fold. Mô hình cuối được chọn theo kết quả xác thực trên train; threshold được xác định từ dự đoán out-of-fold và giữ cố định khi đánh giá holdout. Feature importance của XGBoost được xem cùng permutation importance, không diễn giải điểm importance như quan hệ nhân quả.
