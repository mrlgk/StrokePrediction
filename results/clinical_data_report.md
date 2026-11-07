# Báo cáo dữ liệu theo ngưỡng y khoa

Tạo bởi `scripts/clinical_data_report.py`. Ngưỡng định nghĩa trong `src/preprocessing.py`
(glucose: CDC/ADA; BMI: CDC). Các nhóm chỉ mang tính tham chiếu, không phải chẩn đoán:
dataset không ghi loại xét nghiệm glucose hay thời điểm lấy mẫu.

## Kiểm tra giá trị ngoài khoảng hợp lý
- Số giá trị bị đổi thành NaN: glucose = 0, BMI = 0.
- Dữ liệu sau làm sạch giống dữ liệu gốc: có.
- Ma trận sau tiền xử lý giống nhau: có.

## Tỷ lệ stroke theo nhóm glucose
| Nhóm glucose (mg/dL) | Số ca | Số ca stroke | Tỷ lệ stroke |
|---|---|---|---|
| Hạ đường huyết (54–<70) | 754 | 27 | 3,6% |
| Bình thường (70–<100) | 2377 | 85 | 3,6% |
| Tiền đái tháo đường (100–<126) | 998 | 37 | 3,7% |
| Mức đái tháo đường (≥126) | 981 | 100 | 10,2% |

## Tỷ lệ stroke theo nhóm BMI (người lớn từ 20 tuổi dùng ngưỡng CDC)
| Nhóm BMI (kg/m²) | Số ca | Số ca stroke | Tỷ lệ stroke |
|---|---|---|---|
| Thiếu cân (<18,5) | 36 | 1 | 2,8% |
| Bình thường (18,5–<25) | 805 | 35 | 4,3% |
| Thừa cân (25–<30) | 1297 | 75 | 5,8% |
| Béo phì (≥30) | 1827 | 97 | 5,3% |
| Dưới 20 tuổi (cần percentile) | 944 | 1 | 0,1% |
| Thiếu | 201 | 40 | 19,9% |

Nhóm "Dưới 20 tuổi" không xếp theo ngưỡng người lớn vì BMI trẻ em phải đọc theo percentile tuổi/giới.

## Hồ sơ nhóm thiếu BMI (201 ca, 3,9%)
| Đặc điểm | Nhóm | Số ca thiếu BMI | Tổng | Tỷ lệ thiếu |
|---|---|---|---|---|
| Tuổi | (0, 18] | 21 | 916 | 2,3% |
| Tuổi | (18, 40] | 39 | 1328 | 2,9% |
| Tuổi | (40, 60] | 54 | 1562 | 3,5% |
| Tuổi | (60, 80] | 86 | 1188 | 7,2% |
| Tuổi | (80, 120] | 1 | 116 | 0,9% |
| Stroke | 0 | 161 | 4861 | 3,3% |
| Stroke | 1 | 40 | 249 | 16,1% |
| Tăng huyết áp | 0 | 154 | 4612 | 3,3% |
| Tăng huyết áp | 1 | 47 | 498 | 9,4% |
| Bệnh tim | 0 | 168 | 4834 | 3,5% |
| Bệnh tim | 1 | 33 | 276 | 12,0% |
| Hút thuốc | Unknown | 61 | 1544 | 4,0% |
| Hút thuốc | formerly smoked | 48 | 885 | 5,4% |
| Hút thuốc | never smoked | 40 | 1892 | 2,1% |
| Hút thuốc | smokes | 52 | 789 | 6,6% |

Thiếu BMI không ngẫu nhiên: tập trung ở người lớn tuổi, người từng bị stroke và người có
bệnh nền. Dataset không ghi lý do thiếu; đây là mối liên quan, không phải nguyên nhân.
