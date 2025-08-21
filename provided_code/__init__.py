# === Bộ import chính của dự án OpenKBP ===

from .data_loader import DataLoader  
# 👉 Lớp chịu trách nhiệm load ảnh CT, mask và dose từ các thư mục bệnh nhân.
#    Tự động chia batch cho huấn luyện và dự đoán.

from .dose_evaluation_class import DoseEvaluator  
# 👉 Lớp tính toán các chỉ số đánh giá chất lượng mô hình dự đoán dose (DVH score, Dose score).

from .network_functions import PredictionModel  
# 👉 Lớp huấn luyện và dự đoán liều từ ảnh CT + mask.
#    Kế thừa kiến trúc mạng từ network_architectures.py.

from .utils import get_paths  
# 👉 Hàm tiện ích giúp lấy danh sách đường dẫn bệnh nhân từ thư mục dữ liệu.
#    Ví dụ: get_paths('provided-data/train-pats') trả về list các folder pt_001, pt_002,...
