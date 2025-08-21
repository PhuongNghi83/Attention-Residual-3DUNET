import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Hàm đọc dữ liệu từ 1 file trong tập OpenKBP
def load_file(file_path: Path) -> Union[NDArray, Dict[str, NDArray]]:
    """
    Đọc file từ OpenKBP dataset.
    Trả về:
        - Nếu là file voxel_dimensions.txt → ndarray
        - Nếu là mask (giá trị rỗng) → mảng index
        - Nếu là sparse matrix (vd: dose.csv) → dict chứa data và index
    """
    if file_path.stem == "voxel_dimensions":
        return np.loadtxt(file_path)

    loaded_file_df = pd.read_csv(file_path, index_col=0)
    if loaded_file_df.isnull().values.any():  # Dữ liệu là mask (chứa giá trị rỗng)
        loaded_file = np.array(loaded_file_df.index).squeeze()
    else:  # Dữ liệu là ma trận thưa (sparse matrix)
        loaded_file = {"indices": loaded_file_df.index.values, "data": loaded_file_df.data.values}

    return loaded_file

# Hàm lấy tất cả đường dẫn file trong thư mục
def get_paths(directory_path: Path, extension: Optional[str] = None) -> list[Path]:
    """
    Lấy danh sách đường dẫn đến các file trong thư mục `directory_path`.
    Nếu truyền thêm `extension`, chỉ lấy các file có đuôi mở rộng đó.
    """
    all_paths = []

    if not directory_path.is_dir():
        pass
    elif extension is None:
        dir_list = os.listdir(directory_path)
        for name in dir_list:
            if "." != name[0]:  # Bỏ qua file ẩn
                all_paths.append(directory_path / str(name))
    else:
        data_root = Path(directory_path)
        for file_path in data_root.glob("*.{}".format(extension)):
            file_path = Path(file_path)
            if "." != file_path.stem[0]:
                all_paths.append(file_path)

    return all_paths

# Chuyển mảng 3D thành vector thưa (sparse vector)
def sparse_vector_function(x, indices=None) -> dict[str, NDArray]:
    """
    Chuyển 1 tensor thành vector thưa, chỉ lưu lại phần tử khác 0 và index của chúng.
    Dùng để lưu dose dạng nén.
    """
    if indices is None:
        y = {"data": x[x > 0], "indices": np.nonzero(x.flatten())[-1]}
    else:
        y = {"data": x[x > 0], "indices": indices[x > 0]}
    return y
