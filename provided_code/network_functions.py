# ===== THƯ VIỆN =====
import os
import numpy as np
import pandas as pd
from typing import Optional
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from pathlib import Path

from provided_code.network_architectures import DefineDoseFromCT
from provided_code.batch import DataBatch
from provided_code.data_loader import DataLoader

# ===== TIỆN ÍCH: CHUYỂN MẢNG VỌXEL Dữ LIỆU SANG DẠNG THƯA =====
def sparse_vector_function(volume: np.ndarray) -> dict:
    indices = np.where(volume.flatten() > 0)
    data = volume.flatten()[indices]
    return {"indices": np.array(indices).T, "data": data.reshape(-1, 1)}

# ===== HÀM TÍNH MẤT MÁT: MAE ĐƯỢC MÀ TRỎNG THEO MASK & ROI =====
def masked_mae(y_true, y_pred):
    dose_true = y_true[..., 0]
    mask = y_true[..., 1]
    roi_weight = y_true[..., 2]
    error = tf.abs(dose_true - tf.squeeze(y_pred, axis=-1)) * mask * roi_weight
    return tf.reduce_sum(error) / (tf.reduce_sum(mask * roi_weight) + K.epsilon())

# ===== MẤT MÁT: DMAX (giảm liều đỉnh cao) =====
def dmax_loss(y_true, y_pred):
    dose_true = y_true[..., 0]
    dose_pred = tf.squeeze(y_pred, axis=-1)
    return tf.reduce_mean(tf.abs(tf.reduce_max(dose_true, axis=[1,2,3]) - tf.reduce_max(dose_pred, axis=[1,2,3])))

# ===== MẤT MÁT: D60 KHÔNG DÙNG TFP =====
def d60_loss(y_true, y_pred):
    dose_true = y_true[..., 0]
    dose_pred = tf.squeeze(y_pred, axis=-1)
    batch_size = tf.shape(dose_true)[0]
    num_voxels = tf.reduce_prod(tf.shape(dose_true)[1:])
    dose_true_flat = tf.reshape(dose_true, (batch_size, num_voxels))
    dose_pred_flat = tf.reshape(dose_pred, (batch_size, num_voxels))
    dose_true_sorted = tf.sort(dose_true_flat, axis=-1)
    dose_pred_sorted = tf.sort(dose_pred_flat, axis=-1)
    index = tf.cast(0.4 * tf.cast(num_voxels - 1, tf.float32), tf.int32)
    d60_true = dose_true_sorted[:, index]
    d60_pred = dose_pred_sorted[:, index]
    return tf.reduce_mean(tf.abs(d60_true - d60_pred))

# ===== MẤT MÁT: VÙNG LIỀU THẤP < 30Gy =====
def low_dose_region_loss(y_true, y_pred):
    dose_true = y_true[..., 0]
    dose_pred = tf.squeeze(y_pred, axis=-1)
    mask = y_true[..., 1]
    low_dose_mask = tf.cast(dose_true < 30.0, tf.float32)
    combined_mask = mask * low_dose_mask
    error = tf.abs(dose_true - dose_pred) * combined_mask
    return tf.reduce_sum(error) / (tf.reduce_sum(combined_mask) + K.epsilon())

# ===== TỔNG HỢP NHIỀU HÀM MẤT MÁT =====
def combined_loss(y_true, y_pred, alpha=1.0, beta=0.3, gamma=0.7, delta=0.8):
    return (
        alpha * masked_mae(y_true, y_pred)
        + beta * dmax_loss(y_true, y_pred)
        + gamma * d60_loss(y_true, y_pred)
        + delta * low_dose_region_loss(y_true, y_pred)
    )

# ===== Lớp DÙNG ĐỂ HUẤN LUYỆN MÔ HÌNH DỰ ĐOÁN =====
class PredictionModel(DefineDoseFromCT):
    def __init__(self, data_loader: DataLoader, results_patent_path: Path, model_name: str, stage: str, loss_fn=combined_loss) -> None:
        super().__init__(
            data_shapes=data_loader.data_shapes,
            initial_number_of_filters=16,
            filter_size=(4, 4, 4),
            stride_size=(2, 2, 2),
            gen_optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999),
        )
        self.loss_fn = loss_fn
        self.generator = None
        self.model_name = model_name
        self.data_loader = data_loader
        self.full_roi_list = data_loader.full_roi_list
        self.current_epoch = 0
        self.last_epoch = 10

        model_results_path = results_patent_path / model_name
        self.model_dir = model_results_path / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir = model_results_path / f"{stage}-predictions"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.model_path_template = self.model_dir / "epoch_"

    def train_model(self, epochs: int = 200, save_frequency: int = 5, keep_model_history: int = 2) -> None:
        self._set_epoch_start()
        self.last_epoch = epochs
        self.initialize_networks()
        if self.current_epoch == epochs:
            print(f" Mô hình đã được huấn luyện đủ {epochs} epoch.")
            return

        self.data_loader.set_mode("training_model")
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            print(f"\n Bắt đầu epoch {self.current_epoch}")
            self.data_loader.shuffle_data()

            for idx, batch in enumerate(self.data_loader.get_batches()):
                roi_weight = np.ones_like(batch.possible_dose_mask)

                for roi_name, weight in {
                    "PTV70": 3.0,
                    "PTV63": 2.5,
                    "PTV56": 2.0,
                    "Brainstem": 1.5,
                    "SpinalCord": 1.5
                }.items():
                    idx = batch.get_index_structure_from_structure(roi_name)
                    if idx is not None:
                        roi_mask = batch.structure_masks[..., idx:idx+1]
                        roi_weight += roi_mask * (weight - 1.0)

                y_true_combined = np.concatenate([batch.dose, batch.possible_dose_mask, roi_weight], axis=-1)
                model_loss = self.generator.train_on_batch([batch.ct, batch.structure_masks], y_true_combined)
                print(f" Tổn thất tại epoch {self.current_epoch}, batch {idx}: {model_loss:.3f}")

            self.manage_model_storage(save_frequency, keep_model_history)

    def _set_epoch_start(self) -> None:
        all_model_paths = list(self.model_dir.glob("*.h5"))
        for model_path in all_model_paths:
            *_, epoch_number = model_path.stem.split("epoch_")
            if epoch_number.isdigit():
                self.current_epoch = max(self.current_epoch, int(epoch_number))

    def initialize_networks(self) -> None:
        if self.current_epoch >= 1:
            self.generator = load_model(
                self._get_generator_path(self.current_epoch),
                custom_objects={'masked_mae': self.loss_fn, 'combined_loss': self.loss_fn}
            )
        else:
            self.generator = self.define_generator()
            self.generator.compile(optimizer=Adam(learning_rate=0.0002), loss=self.loss_fn)

    def manage_model_storage(self, save_frequency: int = 1, keep_model_history: Optional[int] = None) -> None:
        effective_epoch_number = self.current_epoch + 1
        if 0 < np.mod(effective_epoch_number, save_frequency) and effective_epoch_number != self.last_epoch:
            return

        epoch_to_overwrite = effective_epoch_number - keep_model_history * (save_frequency or float("inf"))
        if epoch_to_overwrite >= 0:
            initial_model_path = self._get_generator_path(epoch_to_overwrite)
            self.generator.save(initial_model_path)
            os.rename(initial_model_path, self._get_generator_path(effective_epoch_number))
        else:
            self.generator.save(self._get_generator_path(effective_epoch_number))

    def _get_generator_path(self, epoch: Optional[int] = None) -> Path:
        epoch = epoch or self.current_epoch
        return self.model_dir / f"epoch_{epoch}.h5"

    def predict_dose(self, epoch: int = 1) -> None:
        self.generator = load_model(
            self._get_generator_path(epoch),
            custom_objects={'masked_mae': self.loss_fn, 'combined_loss': self.loss_fn}
        )
        os.makedirs(self.prediction_dir, exist_ok=True)
        self.data_loader.set_mode("dose_prediction")

        print(" Đang tiến hành dự đoán phân bố liều...")
        for batch in self.data_loader.get_batches():
            dose_pred = self.generator.predict([batch.ct, batch.structure_masks])
            dose_pred = dose_pred * batch.possible_dose_mask
            dose_pred = np.squeeze(dose_pred)
            dose_to_save = sparse_vector_function(dose_pred)
            dose_df = pd.DataFrame(data=dose_to_save["data"].squeeze(), index=dose_to_save["indices"].squeeze(), columns=["data"])
            (patient_id,) = batch.patient_list
            dose_df.to_csv(f"{self.prediction_dir}/{patient_id}.csv")
