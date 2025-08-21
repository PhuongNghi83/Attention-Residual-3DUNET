from typing import Union
from numpy.typing import NDArray

class DataShapes:
    """
    Lớp định nghĩa kích thước tensor (shape) cho từng loại dữ liệu trong mô hình.
    Dùng để khởi tạo batch với đúng kích thước đầu vào/đầu ra.
    """

    def __init__(self, num_rois):
        self.num_rois = num_rois  # Số lượng cấu trúc (ROI)
        self.patient_shape = (128, 128, 128)  # Kích thước khối 3D của mỗi bệnh nhân (Z, Y, X)

    @property
    def dose(self) -> tuple[int, int, int, int]:
        """Shape của liều thực tế: (128,128,128,1)"""
        return self.patient_shape + (1,)

    @property
    def predicted_dose(self) -> tuple[int, int, int, int]:
        """Shape của liều dự đoán (giống với dose thực)"""
        return self.dose

    @property
    def ct(self) -> tuple[int, int, int, int]:
        """Shape của ảnh CT: 1 kênh xám"""
        return self.patient_shape + (1,)

    @property
    def structure_masks(self) -> tuple[int, int, int, int]:
        """Shape của mask các cấu trúc ROI: mỗi kênh ứng với 1 ROI"""
        return self.patient_shape + (self.num_rois,)

    @property
    def possible_dose_mask(self) -> tuple[int, int, int, int]:
        """Mask nhị phân cho biết vùng có thể nhận liều"""
        return self.patient_shape + (1,)

    @property
    def voxel_dimensions(self) -> tuple[float]:
        """Kích thước voxel (đơn vị mm), chỉ gồm 3 giá trị: dx, dy, dz"""
        return tuple((3,))

    def from_data_names(self, data_names: list[str]) -> dict[str, Union[NDArray, tuple[float]]]:
        """
        Từ danh sách tên dữ liệu, trả về dict chứa shape tương ứng.
        Dùng để khởi tạo batch trống với đúng shape.
        """
        data_shapes = {}
        for name in data_names:
            data_shapes[name] = getattr(self, name)  # Gọi thuộc tính tương ứng (vd: self.ct)
        return data_shapes
