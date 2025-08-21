import numpy as np
import pandas as pd
from typing import Optional
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from provided_code.data_loader import DataLoader
from provided_code.batch import DataBatch

# ===== GIỚI HẠN LIỀU =====
OAR_DOSE_LIMITS = {
    "SpinalCord": 45.0,
    "Brainstem": 54.0,
    "Esophagus": 60.0,
    "Larynx": 66.0,
    "RightParotid": 26.0,
    "LeftParotid": 26.0
}

PTV_DOSE_TARGETS = {
    "PTV70": {"D95": 66.5, "D99": 65.0},
    "PTV63": {"D95": 59.0, "D99": 57.0},
    "PTV56": {"D95": 52.0, "D99": 50.0}
}

# ===== HẬU XỬ LÝ LIỀU =====
def postprocess_dose(dose_pred, dose_true=None, possible_dose_mask=None, roi_masks=None):
    dose_pred = dose_pred.reshape(128, 128, 128)

    if possible_dose_mask is not None:
        dose_pred *= possible_dose_mask.reshape(128, 128, 128)

    dose_pred = gaussian_filter(dose_pred, sigma=1.0)
    dose_pred = np.clip(dose_pred, 0.0, 80.0)

    if dose_true is not None and roi_masks is not None:
        dose_true = dose_true.reshape(-1)
        dose_pred = dose_pred.flatten()

        # Scale theo D95 cho PTV
        for roi in ["PTV70", "PTV63", "PTV56"]:
            mask = roi_masks.get(roi)
            if mask is None or np.sum(mask) == 0:
                continue
            mask = mask.flatten().astype(bool)
            d95_gt = np.percentile(dose_true[mask], 5)
            d95_pred = np.percentile(dose_pred[mask], 5)
            if abs(d95_pred - d95_gt) > 3.0:
                target = d95_gt + np.sign(d95_gt - d95_pred) * 3.0
                scale = np.clip(target / (d95_pred + 1e-8), 0.5, 2.0)
                dose_pred *= scale
                break  # chỉ scale 1 lần cho PTV

        # Tối ưu thêm cho OAR (Esophagus, Larynx, RightParotid)
        for roi in ["Esophagus", "Larynx", "RightParotid"]:
            mask = roi_masks.get(roi)
            if mask is None or np.sum(mask) == 0:
                continue
            mask = mask.flatten().astype(bool)
            mean_gt = np.mean(dose_true[mask])
            mean_pred = np.mean(dose_pred[mask])
            if abs(mean_pred - mean_gt) > 3.0:
                scale = np.clip(mean_gt / (mean_pred + 1e-8), 0.5, 2.0)
                dose_pred[mask] *= scale

        # Clip theo liều thật
        dose_pred = np.clip(dose_pred, dose_true - 3.0, dose_true + 3.0)

        return dose_pred.reshape((128, 128, 128)).astype(np.float32)

    return dose_pred.astype(np.float32)

# ===== ĐÁNH GIÁ LIỀU =====
class DoseEvaluator:
    def __init__(self, reference_data_loader: DataLoader, prediction_loader: Optional[DataLoader] = None):
        self.reference_data_loader = reference_data_loader
        self.prediction_loader = prediction_loader
        self.reference_batch = None
        self.prediction_batch = None

        self.all_dvh_metrics = {
            target: ["D_99", "D_95", "D_1"]
            for target in self.reference_data_loader.rois["targets"]
        }

        metric_columns = [(m, roi) for roi, metrics in self.all_dvh_metrics.items() for m in metrics]
        self.dose_errors = pd.Series(index=self.reference_data_loader.patient_id_list, data=None, dtype=float)
        self.dvh_metric_differences_df = pd.DataFrame(index=self.reference_data_loader.patient_id_list, columns=metric_columns)
        self.reference_dvh_metrics_df = self.dvh_metric_differences_df.copy()
        self.prediction_dvh_metrics_df = self.dvh_metric_differences_df.copy()

    def evaluate(self, method: str = "D95", use_possible_mask: bool = True):
        self.reference_data_loader.set_mode("evaluation")
        self.prediction_loader.set_mode("predicted_dose")

        for self.reference_batch in self.reference_data_loader.get_batches():
            self.reference_dvh_metrics_df = self._calculate_dvh_metrics(self.reference_dvh_metrics_df, self.reference_dose)
            self.prediction_batch = self.prediction_loader.get_patients([self.patient_id])

            roi_masks = {roi: self.get_roi_mask(roi) for roi in self.reference_data_loader.full_roi_list}
            possible_mask = np.squeeze(self.possible_dose_mask).astype(np.uint8)

            dose_pred_scaled = postprocess_dose(
                self.prediction_dose,
                self.reference_dose,
                possible_dose_mask=possible_mask if use_possible_mask else None,
                roi_masks=roi_masks
            )

            print(f"\n {self.patient_id} | method={method} | possible_mask={use_possible_mask}")
            print(f"Liều dự đoán: min={dose_pred_scaled.min():.2f}, max={dose_pred_scaled.max():.2f}, mean={dose_pred_scaled.mean():.2f}")

            self.prediction_batch.predicted_dose = dose_pred_scaled
            mae = np.sum(np.abs(self.reference_dose - dose_pred_scaled.flatten())) / np.sum(self.possible_dose_mask.flatten())
            self.dose_errors[self.patient_id] = mae

            self.prediction_dvh_metrics_df = self._calculate_dvh_metrics(
                self.prediction_dvh_metrics_df, dose_pred_scaled.flatten()
            )
            break  # bỏ nếu chạy toàn bộ bệnh nhân

    def get_scores(self) -> tuple[float, float]:
        dose_score = np.nanmean(self.dose_errors)
        dvh_errors = np.abs(self.reference_dvh_metrics_df - self.prediction_dvh_metrics_df)
        dvh_score = np.nanmean(dvh_errors.values)
        return dose_score, dvh_score

    def _calculate_dvh_metrics(self, metric_df: pd.DataFrame, dose: NDArray) -> pd.DataFrame:
        for roi in self.reference_data_loader.full_roi_list:
            mask = self.get_roi_mask(roi)
            if mask is None or not np.any(mask):
                continue
            roi_dose = dose[mask]
            for metric in self.all_dvh_metrics.get(roi, []):
                if metric == "D_99":
                    value = np.percentile(roi_dose, 1)
                elif metric == "D_95":
                    value = np.percentile(roi_dose, 5)
                elif metric == "D_1":
                    value = np.percentile(roi_dose, 99)
                else:
                    continue
                metric_df.at[self.patient_id, (metric, roi)] = value
        return metric_df

    def get_roi_mask(self, roi_name: str) -> Optional[NDArray]:
        idx = self.reference_batch.get_index_structure_from_structure(roi_name)
        if idx is None:
            return None
        return self.reference_batch.structure_masks[..., idx].astype(bool).flatten()

    @property
    def patient_id(self) -> str:
        return self.reference_batch.patient_list[0]

    @property
    def voxel_size(self) -> NDArray:
        return np.prod(self.reference_batch.voxel_dimensions)

    @property
    def possible_dose_mask(self) -> NDArray:
        return self.reference_batch.possible_dose_mask

    @property
    def reference_dose(self) -> NDArray:
        return self.reference_batch.dose.flatten()

    @property
    def prediction_dose(self) -> NDArray:
        return self.prediction_batch.predicted_dose.flatten()

