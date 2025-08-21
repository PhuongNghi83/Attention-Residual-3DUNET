from __future__ import annotations  # Cho phép tham chiếu lớp chính nó trong chú thích kiểu dữ liệu (Python 3.7+)

from pathlib import Path  # Làm việc với đường dẫn file/thư mục 
from typing import Optional  # Kiểu dữ liệu có thể là None (ví dụ: Optional[int])
import numpy as np  # Thư viện xử lý mảng số 
from numpy.typing import NDArray  # Kiểu dữ liệu mảng nhiều chiều (numpy array)


class DataBatch:
    """
    Lớp chứa dữ liệu của 1 batch (mini-batch) dùng trong huấn luyện hoặc dự đoán.
    Mỗi thuộc tính lưu trữ thông tin tương ứng của batch (CT, mask, dose, ...).
    """

    def __init__(
        self,
        dose: Optional[NDArray] = None,  # Liều thực tế
        predicted_dose: Optional[NDArray] = None,  # Liều dự đoán
        ct: Optional[NDArray] = None,  # Ảnh CT
        structure_masks: Optional[NDArray] = None,  # Mask của các cấu trúc (PTV, OAR,...)
        structure_mask_names: Optional[list[str]] = None,  # Tên tương ứng với mỗi mask
        possible_dose_mask: Optional[NDArray] = None,  # Mask vùng có thể nhận liều
        voxel_dimensions: Optional[NDArray] = None,  # Kích thước mỗi voxel
        patient_list: Optional[list[str]] = None,  # Danh sách ID bệnh nhân trong batch
        patient_path_list: Optional[list[Path]] = None,  # Danh sách đường dẫn bệnh nhân
    ):
        self.dose = dose #Gán dữ liệu 
        self.predicted_dose = predicted_dose
        self.ct = ct
        self.structure_masks = structure_masks
        self.structure_mask_names = structure_mask_names
        self.possible_dose_mask = possible_dose_mask
        self.voxel_dimensions = voxel_dimensions
        self.patient_list = patient_list
        self.patient_path = patient_path_list

    @classmethod
    def initialize_from_required_data(cls, data_dimensions: dict[str, NDArray], batch_size: int) -> DataBatch:
        """
        Tạo mới 1 batch với kích thước cố định và khởi tạo toàn bộ bằng 0.
        Dùng khi khởi tạo batch trống trước khi nạp dữ liệu.
        """
        attribute_values = {}
        for data, dimensions in data_dimensions.items():
            batch_data_dimensions = (batch_size, *dimensions)  # VD: (2, 128, 128, 128)
            attribute_values[data] = np.zeros(batch_data_dimensions)
        return cls(**attribute_values)

    def set_values(self, data_name: str, batch_index: int, values: NDArray):
        """
        Gán giá trị cho batch tại chỉ số cụ thể.
        Ví dụ: set ảnh CT của bệnh nhân thứ i trong batch.
        """
        getattr(self, data_name)[batch_index] = values

    def get_index_structure_from_structure(self, structure_name: str):
        """
        Lấy vị trí chỉ số trong danh sách tên mask (VD: index của 'PTV70').
        """
        return self.structure_mask_names.index(structure_name)