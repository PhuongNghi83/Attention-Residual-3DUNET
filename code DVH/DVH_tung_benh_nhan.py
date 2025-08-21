# ===== THƯ VIỆN =====
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from numpy.typing import NDArray
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

from provided_code.data_loader import DataLoader          # Dùng để load CT, mask, dose
from provided_code.dose_evaluation_class import DoseEvaluator  # Đánh giá liều thực tế và dự đoán

# ===== HÀM TÍNH ĐƯỜNG CONG DVH =====
def compute_dvh(dose: np.ndarray, mask: np.ndarray, bins: int = 100, max_dose: float = 80):
    values = dose[mask == 1]  # Lấy giá trị liều tại vùng mask
    if values.size == 0:
        return None, None
    bin_edges = np.linspace(0, max_dose, bins)  # Chia liều thành các khoảng
    hist, _ = np.histogram(values, bins=bin_edges)
    cdf = np.cumsum(hist[::-1])[::-1] / len(values) * 100  # Tính phân phối tích lũy ngược
    return bin_edges[:-1], cdf  # Trả về trục X và Y cho đồ thị

# ===== IN BẢNG CHỈ SỐ LIỀU DỰ ĐOÁN VÀ THỰC TẾ =====
def print_full_dose_table(dvh_dict: dict):
    cols = ["D95", "D60", "D20", "D2", "mean"]
    header = f"{'Cơ quan/ROI':<15} |" + "".join([f" {col}_GT | {col}_Pred ||" for col in cols])
    print("\nBẢNG SO SÁNH CHỈ SỐ LIỀU (GT vs Dự đoán):")
    print(header)
    print("-" * len(header))

    for roi, metrics in dvh_dict.items():
        row = f"{roi:<15} |"
        for col in cols:
            gt = metrics.get(f"{col}_GT", 0)
            pr = metrics.get(f"{col}_Pred", 0)
            row += f" {gt:6.2f} | {pr:7.2f} ||"
        print(row)

# ===== HÀM VẼ ĐƯỜNG CONG DVH & IN BẢNG CHỈ SỐ =====
def plot_dvh(patient_id: str, ref_path: Path, pred_path: Path):
    from provided_code.dose_evaluation_class import postprocess_dose

    # Tải dữ liệu thực tế và dự đoán
    ref_loader = DataLoader([ref_path])
    pred_loader = DataLoader([pred_path])
    evaluator = DoseEvaluator(reference_data_loader=ref_loader, prediction_loader=pred_loader)
    evaluator.evaluate()

    dose_true = evaluator.reference_dose
    dose_pred = evaluator.prediction_dose.copy()

    # Lấy mặt nạ của các cơ quan/ROI
    roi_masks = {roi: evaluator.get_roi_mask(roi) for roi in evaluator.reference_data_loader.full_roi_list}

    # Vẽ đồ thị DVH
    plt.figure(figsize=(10, 6))
    for i, (roi, mask) in enumerate(roi_masks.items()):
        if mask is None or np.sum(mask) == 0:
            continue

        x_gt, y_gt = compute_dvh(dose_true, mask)
        x_pr, y_pr = compute_dvh(dose_pred, mask)

        if y_gt is not None:
            y_gt = gaussian_filter1d(y_gt, sigma=1.2)
        if y_pr is not None:
            y_pr = gaussian_filter1d(y_pr, sigma=1.2)

        if x_gt is not None and y_gt is not None:
            plt.plot(x_gt, y_gt, label=f"{roi} GT", color=f"C{i}")
        if x_pr is not None and y_pr is not None:
            plt.plot(x_pr, y_pr, linestyle="--", label=f"{roi} Dự đoán", color=f"C{i}")

    plt.title(f"Đường cong DVH - Bệnh nhân {patient_id} ")
    plt.xlabel("Liều (Gy)")
    plt.ylabel("Tỉ lệ thể tích (%)")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    plt.tight_layout()

    # Tính các chỉ số D95, D60, D20, D2 và trung bình
    dose_metrics = {}
    for roi, mask in roi_masks.items():
        if mask is None or np.sum(mask) == 0:
            continue

        gt_vals = dose_true[mask]
        pr_vals = dose_pred[mask]

        dose_metrics[roi] = {
            "D95_GT": np.percentile(gt_vals, 5),
            "D60_GT": np.percentile(gt_vals, 40),
            "D20_GT": np.percentile(gt_vals, 80),
            "D2_GT": np.percentile(gt_vals, 98),
            "mean_GT": np.mean(gt_vals),
            "D95_Pred": np.percentile(pr_vals, 5),
            "D60_Pred": np.percentile(pr_vals, 40),
            "D20_Pred": np.percentile(pr_vals, 80),
            "D2_Pred": np.percentile(pr_vals, 98),
            "mean_Pred": np.mean(pr_vals),
        }

    # In bảng chỉ số so sánh
    print_full_dose_table(dose_metrics)

    plt.show()

# ===== CHẠY VỚI MỘT BỆNH NHÂN =====
if __name__ == "__main__":
    patient_id = "pt_227"
    ref_path = Path(f"provided-data/validation-pats/{patient_id}")
    pred_path = Path(f"results/baseline/validation-predictions/{patient_id}.csv")
    plot_dvh(patient_id, ref_path, pred_path)
