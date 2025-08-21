from pathlib import Path  # Làm việc với đường dẫn file
from typing import Dict, Iterator, List, Optional, Union

import numpy as np  # Xử lý mảng
from more_itertools import windowed  # Tạo các batch từ danh sách (trượt cửa sổ)
from numpy.typing import NDArray
from tqdm import tqdm  # Hiển thị thanh tiến trình

from provided_code.batch import DataBatch  # Lớp chứa dữ liệu của từng batch
from provided_code.data_shapes import DataShapes  # Cấu hình kích thước dữ liệu
from provided_code.utils import get_paths, load_file  # Hàm tiện ích load file


class DataLoader:
    """Lớp chịu trách nhiệm load và xử lý dữ liệu OpenKBP thành dạng batch để dùng cho mô hình."""

    def __init__(self, patient_paths: List[Path], batch_size: int = 2):
        self.patient_paths = patient_paths  # Danh sách đường dẫn từng bệnh nhân
        self.batch_size = batch_size

        # Map tên bệnh nhân sang đường dẫn
        self.paths_by_patient_id = {patient_path.stem: patient_path for patient_path in self.patient_paths}
        self.required_files: Optional[Dict] = None  # Các file cần load tương ứng với chế độ
        self.mode_name: Optional[str] = None  # Tên chế độ hiện tại (training, prediction,...)

        # Các ROI mặc định (theo cấu trúc của OpenKBP)
        self.rois = dict(
            oars=["Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Esophagus", "Larynx", "Mandible"],
            targets=["PTV56", "PTV63", "PTV70"],
        )
        self.full_roi_list = sum(map(list, self.rois.values()), [])  # Gộp thành danh sách các ROI
        self.num_rois = len(self.full_roi_list)
        self.data_shapes = DataShapes(self.num_rois)  # Tạo thông tin shape dữ liệu từ số ROI

    @property
    def patient_id_list(self) -> List[str]:
        return list(self.paths_by_patient_id.keys())  # Trả về danh sách ID bệnh nhân

    def get_batches(self) -> Iterator[DataBatch]:
        # Tạo các batch từ danh sách bệnh nhân
        batches = windowed(self.patient_paths, n=self.batch_size, step=self.batch_size)
        complete_batches = (batch for batch in batches if None not in batch)  # Bỏ batch thiếu
        for batch_paths in tqdm(complete_batches):  # Hiển thị tiến trình
            yield self.prepare_data(batch_paths)  # Trả về batch dữ liệu

    def get_patients(self, patient_list: List[str]) -> DataBatch:
        # Load batch cụ thể từ danh sách tên bệnh nhân
        file_paths_to_load = [self.paths_by_patient_id[patient] for patient in patient_list]
        return self.prepare_data(file_paths_to_load)

    def set_mode(self, mode: str) -> None:
        """Thiết lập chế độ xử lý: huấn luyện, đánh giá, dự đoán..."""
        self.mode_name = mode
        if mode == "training_model":
            required_data = ["dose", "ct", "structure_masks", "possible_dose_mask", "voxel_dimensions"]
        elif mode == "predicted_dose":
            required_data = [mode]
            self._force_batch_size_one()
        elif mode == "evaluation":
            required_data = ["dose", "structure_masks", "possible_dose_mask", "voxel_dimensions"]
            self._force_batch_size_one()
        elif mode == "dose_prediction":
            required_data = ["ct", "structure_masks", "possible_dose_mask", "voxel_dimensions"]
            self._force_batch_size_one()
        else:
            raise ValueError(f"Mode `{mode}` không hợp lệ. Chọn 1 trong: training_model, prediction, evaluation,...")
        self.required_files = self.data_shapes.from_data_names(required_data)

    def _force_batch_size_one(self) -> None:
        """Ép batch size = 1 (bắt buộc với dự đoán hoặc đánh giá từng bệnh nhân)"""
        if self.batch_size != 1:
            self.batch_size = 1
            Warning("Batch size đã chuyển về 1 do chế độ yêu cầu")

    def shuffle_data(self) -> None:
        """Trộn ngẫu nhiên dữ liệu bệnh nhân (tránh overfitting khi huấn luyện)"""
        np.random.shuffle(self.patient_paths)

    def prepare_data(self, file_paths_to_load: List[Path]) -> DataBatch:
        """Tạo batch chứa dữ liệu các bệnh nhân và reshape đúng kích thước"""

        batch_data = DataBatch.initialize_from_required_data(self.required_files, self.batch_size)
        batch_data.patient_list = [patient_path.stem for patient_path in file_paths_to_load]
        batch_data.patient_path_list = file_paths_to_load
        batch_data.structure_mask_names = self.full_roi_list

        # Load từng bệnh nhân trong batch
        for index, patient_path in enumerate(file_paths_to_load):
            raw_data = self.load_data(patient_path)
            for key in self.required_files:
                batch_data.set_values(key, index, self.shape_data(key, raw_data))

        return batch_data

    def load_data(self, path_to_load: Path) -> Union[NDArray, dict[str, NDArray]]:
        """Load dữ liệu gốc từ thư mục bệnh nhân"""
        data = {}
        if path_to_load.is_dir():
            files_to_load = get_paths(path_to_load)
            for file_path in files_to_load:
                is_required = file_path.stem in self.required_files
                is_required_roi = file_path.stem in self.full_roi_list
                if is_required or is_required_roi:
                    data[file_path.stem] = load_file(file_path)
        else:
            data[self.mode_name] = load_file(path_to_load)

        return data

    def shape_data(self, key: str, data: dict) -> NDArray:
        """Chuyển dữ liệu thô sang dạng mảng có kích thước phù hợp với mô hình"""

        shaped_data = np.zeros(self.required_files[key])  # Khởi tạo mảng có shape đúng

        if key == "structure_masks":
            # Với mỗi ROI, đánh dấu pixel thuộc ROI đó bằng giá trị 1
            for roi_idx, roi in enumerate(self.full_roi_list):
                if roi in data.keys():
                    np.put(shaped_data, self.num_rois * data[roi] + roi_idx, int(1))
        elif key == "possible_dose_mask":
            # Mask vùng có thể nhận liều
            np.put(shaped_data, data[key], int(1))
        elif key == "voxel_dimensions":
            # Kích thước voxel (không reshape)
            shaped_data = data[key]
        else:
            # Các dữ liệu dạng sparse: gán value vào index tương ứng
            np.put(shaped_data, data[key]["indices"], data[key]["data"])

        return shaped_data
