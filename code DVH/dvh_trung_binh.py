'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from provided_code.data_loader import DataLoader
from provided_code.dose_evaluation_class import DoseEvaluator

# === THƯ MỤC DỮ LIỆU ===
ref_dir = Path("provided-data/validation-pats")
pred_dir = Path("results/baseline/validation-predictions")

# === DUYỆT TOÀN BỘ BỆNH NHÂN ĐỂ TÍNH CHỈ SỐ ===
patient_ids = [p.name for p in ref_dir.iterdir() if p.is_dir()]
results = []

for patient_id in patient_ids:
    try:
        ref_path = ref_dir / patient_id
        pred_path = pred_dir / f"{patient_id}.csv"

        ref_loader = DataLoader([ref_path])
        pred_loader = DataLoader([pred_path])
        evaluator = DoseEvaluator(reference_data_loader=ref_loader, prediction_loader=pred_loader)
        evaluator.evaluate()

        dose_true = evaluator.reference_dose
        dose_pred = evaluator.prediction_dose.copy()
        roi_masks = {roi: evaluator.get_roi_mask(roi) for roi in evaluator.reference_data_loader.full_roi_list}

        for roi, mask in roi_masks.items():
            if mask is None or np.sum(mask) == 0:
                continue

            gt_vals = dose_true[mask]
            pr_vals = dose_pred[mask]

            results.append({
                "Patient_ID": patient_id,
                "ROI": roi,
                "D95_GT": np.percentile(gt_vals, 5),
                "D99_GT": np.percentile(gt_vals, 1),
                "D60_GT": np.percentile(gt_vals, 40),
                "D20_GT": np.percentile(gt_vals, 80),
                "D2_GT": np.percentile(gt_vals, 98),
                "mean_GT": np.mean(gt_vals),
                "D95_Pred": np.percentile(pr_vals, 5),
                "D99_Pred": np.percentile(pr_vals, 1),
                "D60_Pred": np.percentile(pr_vals, 40),
                "D20_Pred": np.percentile(pr_vals, 80),
                "D2_Pred": np.percentile(pr_vals, 98),
                "mean_Pred": np.mean(pr_vals),
            })
    except Exception as e:
        print(f"[!] Lỗi bệnh nhân {patient_id}: {e}")

# === TÍNH TRUNG BÌNH VÀ ĐỘ LỆCH CHUẨN ===
df = pd.DataFrame(results)

# Thêm D99 vào danh sách metrics
metrics = ["D95", "D99", "D60", "D20", "D2", "mean"]

# Tính MAE cho từng chỉ số
for m in metrics:
    df[f"MAE_{m}"] = np.abs(df[f"{m}_GT"] - df[f"{m}_Pred"])

# Gom theo ROI, tính mean và std
summary = df.groupby("ROI").agg(["mean", "std"]).reset_index()

# Tạo bảng định dạng
formatted = pd.DataFrame()
formatted["ROI"] = summary["ROI"]

# Thêm GT ± std, Pred ± std
for metric in metrics:
    for typ in ["GT", "Pred"]:
        col = f"{metric}_{typ}"
        mean = summary[(col, "mean")].round(2)
        std = summary[(col, "std")].round(2)
        formatted[f"{col} ±"] = mean.astype(str) + " ± " + std.astype(str)

# Thêm MAE ± std
for m in metrics:
    mean = summary[(f"MAE_{m}", "mean")].round(2)
    std = summary[(f"MAE_{m}", "std")].round(2)
    formatted[f"MAE_{m} ±"] = mean.astype(str) + " ± " + std.astype(str)

# Xuất ra Excel
formatted.to_excel("dvh_metrics_all_patients_mean_std.xlsx", index=False)

# === HÀM TÍNH DVH TRUNG BÌNH CHO GT VÀ PRED ===
def compute_avg_dvh_both(df: pd.DataFrame, bins: int = 100, max_dose: float = 80):
    dose_bins = np.linspace(0, max_dose, bins)
    mid_bins = dose_bins[:-1]
    dvh_gt_dict, dvh_pred_dict = {}, {}

    for roi in df["ROI"].unique():
        roi_df = df[df["ROI"] == roi]
        gt_all, pred_all = [], []

        for _, row in roi_df.iterrows():
            try:
                pred_loader = DataLoader([pred_dir / f"{row['Patient_ID']}.csv"])
                ref_loader = DataLoader([ref_dir / row["Patient_ID"]])
                evaluator = DoseEvaluator(reference_data_loader=ref_loader, prediction_loader=pred_loader)
                evaluator.evaluate()

                gt_dose = evaluator.reference_dose
                pred_dose = evaluator.prediction_dose
                mask = evaluator.get_roi_mask(roi)

                if mask is None or np.sum(mask) == 0:
                    continue

                gt_values = gt_dose[mask]
                pred_values = pred_dose[mask]

                gt_hist, _ = np.histogram(gt_values, bins=dose_bins)
                pred_hist, _ = np.histogram(pred_values, bins=dose_bins)

                gt_cdf = np.cumsum(gt_hist[::-1])[::-1] / len(gt_values) * 100
                pred_cdf = np.cumsum(pred_hist[::-1])[::-1] / len(pred_values) * 100

                gt_all.append(gt_cdf)
                pred_all.append(pred_cdf)
            except:
                continue

        if gt_all:
            dvh_gt_dict[roi] = np.mean(gt_all, axis=0)
        if pred_all:
            dvh_pred_dict[roi] = np.mean(pred_all, axis=0)

    return mid_bins, dvh_gt_dict, dvh_pred_dict

# === HÀM VẼ DVH ===
def plot_avg_dvh_gt_pred(mid_bins, dvh_gt, dvh_pred, title, save_path):
    plt.figure(figsize=(10, 6))
    for roi in dvh_gt:
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.plot(mid_bins, dvh_gt[roi], label=f"{roi} - Thực tế", linestyle='-', color=color)
        if roi in dvh_pred:
            plt.plot(mid_bins, dvh_pred[roi], label=f"{roi} - Dự đoán", linestyle='--', color=color)

    plt.xlabel("Liều (Gy)")
    plt.ylabel("Tỷ lệ thể tích còn lại (%)")
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path + ".pdf")
    plt.savefig(save_path + ".png", dpi=300)
    plt.close()

# === DANH SÁCH ROI ===
OAR_ROIs = {"Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Larynx", "Esophagus", "Mandible"}
PTV_ROIs = {"PTV56", "PTV63", "PTV70"}

# === VẼ DVH TRUNG BÌNH CHO OAR ===
mid_bins, dvh_gt_oar, dvh_pred_oar = compute_avg_dvh_both(df[df["ROI"].isin(OAR_ROIs)])
plot_avg_dvh_gt_pred(mid_bins, dvh_gt_oar, dvh_pred_oar, "DVH trung bình cho OAR", "dvh_avg_oar")

# === VẼ DVH TRUNG BÌNH CHO PTV ===
mid_bins, dvh_gt_ptv, dvh_pred_ptv = compute_avg_dvh_both(df[df["ROI"].isin(PTV_ROIs)])
plot_avg_dvh_gt_pred(mid_bins, dvh_gt_ptv, dvh_pred_ptv, "DVH trung bình cho PTV", "dvh_avg_ptv")

print("✅ Đã tính D99 và xuất đầy đủ bảng kết quả DVH có MAE ± std.")'''
'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from provided_code.data_loader import DataLoader
from provided_code.dose_evaluation_class import DoseEvaluator

# ==== Dịch tên ROI sang tiếng Việt ====
ROI_VIETNAMESE = {
    "Brainstem": "Thân não",
    "SpinalCord": "Tủy sống",
    "RightParotid": "Tuyến mang tai phải",
    "LeftParotid": "Tuyến mang tai trái",
    "Larynx": "Thanh quản",
    "Esophagus": "Thực quản",
    "Mandible": "Xương hàm",
    "PTV56": "PTV56",
    "PTV63": "PTV63",
    "PTV70": "PTV70"
}

# ==== Thư mục dữ liệu ====
ref_dir = Path("provided-data/validation-pats")
pred_dir = Path("results/baseline/validation-predictions")

# ==== Tính các chỉ số cho từng bệnh nhân ====
patient_ids = [p.name for p in ref_dir.iterdir() if p.is_dir()]
results = []
results_cache = {}

for patient_id in patient_ids:
    try:
        ref_path = ref_dir / patient_id
        pred_path = pred_dir / f"{patient_id}.csv"
        ref_loader = DataLoader([ref_path])
        pred_loader = DataLoader([pred_path])
        evaluator = DoseEvaluator(ref_loader, pred_loader)
        evaluator.evaluate()

        dose_true = evaluator.reference_dose
        dose_pred = evaluator.prediction_dose.copy()
        roi_masks = {roi: evaluator.get_roi_mask(roi) for roi in ref_loader.full_roi_list}
        results_cache[patient_id] = (dose_true, dose_pred, roi_masks)

        for roi, mask in roi_masks.items():
            if mask is None or np.sum(mask) < 100:
                continue
            gt = dose_true[mask]
            pr = dose_pred[mask]
            results.append({
                "Patient_ID": patient_id,
                "ROI": roi,
                "D95_GT": np.percentile(gt, 5),
                "D99_GT": np.percentile(gt, 1),
                "D2_GT": np.percentile(gt, 98),
                "Dmax_GT": np.max(gt),
                "Dmean_GT": np.mean(gt),
                "D95_Pred": np.percentile(pr, 5),
                "D99_Pred": np.percentile(pr, 1),
                "D2_Pred": np.percentile(pr, 98),
                "Dmax_Pred": np.max(pr),
                "Dmean_Pred": np.mean(pr),
            })
    except Exception as e:
        print(f"[!] Lỗi {patient_id}: {e}")

# ==== Tổng hợp bảng chỉ số tiếng Việt ====
df = pd.DataFrame(results)
metrics = ["D95", "D99", "D2", "Dmax", "Dmean"]
summary = df.groupby("ROI", as_index=False).mean(numeric_only=True)

formatted = pd.DataFrame()
formatted["Cơ quan (ROI)"] = summary["ROI"].map(ROI_VIETNAMESE)

for metric in metrics:
    gt_col = f"{metric}_GT"
    pred_col = f"{metric}_Pred"
    err = (summary[gt_col] - summary[pred_col]).abs().round(2)

    formatted[f"{metric} (Thực tế)"] = summary[gt_col].round(2)
    formatted[f"{metric} (Dự đoán)"] = summary[pred_col].round(2)
    formatted[f"{metric} (Sai số tuyệt đối)"] = err

formatted.to_excel("dvh_metrics_error_detail.xlsx", index=False)
print("Đã lưu: dvh_metrics_error_detail.xlsx")

# ==== Tạo bảng theo định dạng OpenKBP ====
def calculate_summary_table_openkbp(df: pd.DataFrame) -> pd.DataFrame:
    target_metrics = [
        ("PTV70", "D99"),
        ("PTV63", "D99"),
        ("PTV56", "D99"),
        ("Brainstem", "Dmax"),
        ("LeftParotid", "Dmean"),
        ("RightParotid", "Dmean"),
        ("SpinalCord", "Dmax"),
        ("Esophagus", "Dmean"),
        ("Larynx", "Dmean"),
        ("Mandible", "Dmax"),
    ]
    rows = []
    for roi, metric in target_metrics:
        gt_col = f"{metric}_GT"
        pr_col = f"{metric}_Pred"
        roi_df = df[df["ROI"] == roi]
        if gt_col not in roi_df or pr_col not in roi_df:
            continue
        gt_vals = roi_df[gt_col]
        pr_vals = roi_df[pr_col]
        diff_vals = pr_vals - gt_vals
        percent_diff = 100 * diff_vals / gt_vals.replace(0, np.nan)
        row = {
            "Chỉ số": f"{ROI_VIETNAMESE.get(roi, roi)} - {metric}",
            "Thực tế (Gy)": f"{gt_vals.mean():.2f}±{gt_vals.std():.2f}",
            "Dự đoán (Gy)": f"{pr_vals.mean():.2f}±{pr_vals.std():.2f}",
            "Hiệu số (Gy)": f"{diff_vals.mean():.2f}±{diff_vals.std():.2f} ({percent_diff.mean():.2f}%)"
        }
        rows.append(row)
    return pd.DataFrame(rows)

openkbp_table = calculate_summary_table_openkbp(df)
openkbp_table.to_excel("openkbp_style_summary.xlsx", index=False)
print("Đã lưu: openkbp_style_summary.xlsx")

# ==== Tính DVH trung bình ====
def compute_avg_dvh_from_cache(df: pd.DataFrame, bins=100, max_dose=80):
    dose_bins = np.linspace(0, max_dose, bins)
    mid_bins = dose_bins[:-1]
    dvh_gt_dict, dvh_pred_dict = {}, {}
    for roi in df["ROI"].unique():
        roi_df = df[df["ROI"] == roi]
        gt_all, pred_all = [], []
        for _, row in roi_df.iterrows():
            pid = row["Patient_ID"]
            dose_true, dose_pred, roi_masks = results_cache[pid]
            mask = roi_masks.get(roi, None)
            if mask is None or np.sum(mask) == 0:
                continue
            gt = dose_true[mask]
            pr = dose_pred[mask]
            gt_hist, _ = np.histogram(gt, bins=dose_bins)
            pr_hist, _ = np.histogram(pr, bins=dose_bins)
            gt_cdf = np.cumsum(gt_hist[::-1])[::-1] / len(gt) * 100
            pr_cdf = np.cumsum(pr_hist[::-1])[::-1] / len(pr) * 100
            gt_all.append(gt_cdf)
            pred_all.append(pr_cdf)
        if gt_all: dvh_gt_dict[roi] = np.mean(gt_all, axis=0)
        if pred_all: dvh_pred_dict[roi] = np.mean(pred_all, axis=0)
    return mid_bins, dvh_gt_dict, dvh_pred_dict

# ==== Hàm vẽ DVH (không hiển thị Dxx hay đường ngang) ====
def plot_avg_dvh(mid_bins, dvh_gt, dvh_pred, title, save_path, summary_df=None, show_dxx_for_ptv=False):
    plt.figure(figsize=(10, 6))

    for roi in dvh_gt:
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        name_vn = ROI_VIETNAMESE.get(roi, roi)
        plt.plot(mid_bins, dvh_gt[roi], label=f"{name_vn} - Thực tế", linestyle='-', color=color)
        if roi in dvh_pred:
            plt.plot(mid_bins, dvh_pred[roi], label=f"{name_vn} - Dự đoán", linestyle='--', color=color)

    plt.xlabel("Liều (Gy)")
    plt.ylabel("Tỷ lệ thể tích còn lại (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="medium")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path + ".pdf")
    plt.savefig(save_path + ".png", dpi=300)
    plt.close()

# ==== Vẽ DVH ====
OARs = {"Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Larynx", "Esophagus", "Mandible"}
PTVs = {"PTV56", "PTV63", "PTV70"}

mid, gt_oar, pr_oar = compute_avg_dvh_from_cache(df[df["ROI"].isin(OARs)])
plot_avg_dvh(mid, gt_oar, pr_oar, "DVH trung bình cho OAR", "dvh_avg_oar")

mid, gt_ptv, pr_ptv = compute_avg_dvh_from_cache(df[df["ROI"].isin(PTVs)])
plot_avg_dvh(mid, gt_ptv, pr_ptv, "DVH trung bình cho PTV", "dvh_avg_ptv", summary_df=summary, show_dxx_for_ptv=False)

print("Hoàn tất: Đã xuất bảng kết quả và biểu đồ DVH.")'''
'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from provided_code.data_loader import DataLoader
from provided_code.dose_evaluation_class import DoseEvaluator

# ==== Dịch tên ROI sang tiếng Việt ====
ROI_VIETNAMESE = {
    "Brainstem": "Thân não",
    "SpinalCord": "Tủy sống",
    "RightParotid": "Tuyến mang tai phải",
    "LeftParotid": "Tuyến mang tai trái",
    "Larynx": "Thanh quản",
    "Esophagus": "Thực quản",
    "Mandible": "Xương hàm",
    "PTV56": "PTV56",
    "PTV63": "PTV63",
    "PTV70": "PTV70"
}

# ==== Thư mục dữ liệu ====
ref_dir = Path("provided-data/validation-pats")
pred_dir = Path("results/baseline/validation-predictions")

# ==== Tính các chỉ số cho từng bệnh nhân ====
patient_ids = [p.name for p in ref_dir.iterdir() if p.is_dir()]
results = []
results_cache = {}

for patient_id in patient_ids:
    try:
        ref_path = ref_dir / patient_id
        pred_path = pred_dir / f"{patient_id}.csv"
        ref_loader = DataLoader([ref_path])
        pred_loader = DataLoader([pred_path])
        evaluator = DoseEvaluator(ref_loader, pred_loader)
        evaluator.evaluate()

        dose_true = evaluator.reference_dose
        dose_pred = evaluator.prediction_dose.copy()
        roi_masks = {roi: evaluator.get_roi_mask(roi) for roi in ref_loader.full_roi_list}
        results_cache[patient_id] = (dose_true, dose_pred, roi_masks)

        for roi, mask in roi_masks.items():
            if mask is None or np.sum(mask) < 100:
                continue
            gt = dose_true[mask]
            pr = dose_pred[mask]
            results.append({
                "Patient_ID": patient_id,
                "ROI": roi,
                "D99_GT": np.percentile(gt, 1),
                "D60_GT": np.percentile(gt, 40),
                "D20_GT": np.percentile(gt, 80),
                "D2_GT": np.percentile(gt, 98),
                "Dmax_GT": np.max(gt),
                "Dmean_GT": np.mean(gt),
                "D99_Pred": np.percentile(pr, 1),
                "D60_Pred": np.percentile(pr, 40),
                "D20_Pred": np.percentile(pr, 80),
                "D2_Pred": np.percentile(pr, 98),
                "Dmax_Pred": np.max(pr),
                "Dmean_Pred": np.mean(pr),
            })
    except Exception as e:
        print(f"[!] Lỗi {patient_id}: {e}")

# ==== Tổng hợp bảng chỉ số tiếng Việt ====
df = pd.DataFrame(results)
metrics = ["D99", "D60", "D20", "D2", "Dmax", "Dmean"]
summary = df.groupby("ROI", as_index=False).mean(numeric_only=True)

formatted = pd.DataFrame()
formatted["Cơ quan (ROI)"] = summary["ROI"].map(ROI_VIETNAMESE)

for metric in metrics:
    gt_col = f"{metric}_GT"
    pred_col = f"{metric}_Pred"
    err = (summary[gt_col] - summary[pred_col]).abs().round(2)

    formatted[f"{metric} (Thực tế)"] = summary[gt_col].round(2)
    formatted[f"{metric} (Dự đoán)"] = summary[pred_col].round(2)
    formatted[f"{metric} (Sai số tuyệt đối)"] = err

formatted.to_excel("dvh_metrics_error_detail.xlsx", index=False)
print("Đã lưu: dvh_metrics_error_detail.xlsx")

# ==== Tạo bảng theo định dạng OpenKBP (giữ nguyên) ====
def calculate_summary_table_openkbp(df: pd.DataFrame) -> pd.DataFrame:
    target_metrics = [
        ("PTV70", "D99"),
        ("PTV63", "D99"),
        ("PTV56", "D99"),
        ("Brainstem", "Dmax"),
        ("LeftParotid", "Dmean"),
        ("RightParotid", "Dmean"),
        ("SpinalCord", "Dmax"),
        ("Esophagus", "Dmean"),
        ("Larynx", "Dmean"),
        ("Mandible", "Dmax"),
    ]
    rows = []
    for roi, metric in target_metrics:
        gt_col = f"{metric}_GT"
        pr_col = f"{metric}_Pred"
        roi_df = df[df["ROI"] == roi]
        if gt_col not in roi_df or pr_col not in roi_df:
            continue
        gt_vals = roi_df[gt_col]
        pr_vals = roi_df[pr_col]
        diff_vals = pr_vals - gt_vals
        percent_diff = 100 * diff_vals / gt_vals.replace(0, np.nan)
        row = {
            "Chỉ số": f"{ROI_VIETNAMESE.get(roi, roi)} - {metric}",
            "Thực tế (Gy)": f"{gt_vals.mean():.2f}±{gt_vals.std():.2f}",
            "Dự đoán (Gy)": f"{pr_vals.mean():.2f}±{pr_vals.std():.2f}",
            "Hiệu số (Gy)": f"{diff_vals.mean():.2f}±{diff_vals.std():.2f} ({percent_diff.mean():.2f}%)"
        }
        rows.append(row)
    return pd.DataFrame(rows)

openkbp_table = calculate_summary_table_openkbp(df)
openkbp_table.to_excel("openkbp_style_summary.xlsx", index=False)
print("Đã lưu: openkbp_style_summary.xlsx")

# ==== Tính DVH trung bình ====
def compute_avg_dvh_from_cache(df: pd.DataFrame, bins=100, max_dose=80):
    dose_bins = np.linspace(0, max_dose, bins)
    mid_bins = dose_bins[:-1]
    dvh_gt_dict, dvh_pred_dict = {}, {}
    for roi in df["ROI"].unique():
        roi_df = df[df["ROI"] == roi]
        gt_all, pred_all = [], []
        for _, row in roi_df.iterrows():
            pid = row["Patient_ID"]
            dose_true, dose_pred, roi_masks = results_cache[pid]
            mask = roi_masks.get(roi, None)
            if mask is None or np.sum(mask) == 0:
                continue
            gt = dose_true[mask]
            pr = dose_pred[mask]
            gt_hist, _ = np.histogram(gt, bins=dose_bins)
            pr_hist, _ = np.histogram(pr, bins=dose_bins)
            gt_cdf = np.cumsum(gt_hist[::-1])[::-1] / len(gt) * 100
            pr_cdf = np.cumsum(pr_hist[::-1])[::-1] / len(pr) * 100
            gt_all.append(gt_cdf)
            pred_all.append(pr_cdf)
        if gt_all: dvh_gt_dict[roi] = np.mean(gt_all, axis=0)
        if pred_all: dvh_pred_dict[roi] = np.mean(pred_all, axis=0)
    return mid_bins, dvh_gt_dict, dvh_pred_dict

# ==== Hàm vẽ DVH (không hiển thị Dxx hay đường ngang) ====
def plot_avg_dvh(mid_bins, dvh_gt, dvh_pred, title, save_path, summary_df=None, show_dxx_for_ptv=False):
    plt.figure(figsize=(10, 6))

    for roi in dvh_gt:
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        name_vn = ROI_VIETNAMESE.get(roi, roi)
        plt.plot(mid_bins, dvh_gt[roi], label=f"{name_vn} - Thực tế", linestyle='-', color=color)
        if roi in dvh_pred:
            plt.plot(mid_bins, dvh_pred[roi], label=f"{name_vn} - Dự đoán", linestyle='--', color=color)

    plt.xlabel("Liều (Gy)")
    plt.ylabel("Tỷ lệ thể tích còn lại (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="medium")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path + ".pdf")
    plt.savefig(save_path + ".png", dpi=300)
    plt.close()

# ==== Vẽ DVH ====
OARs = {"Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Larynx", "Esophagus", "Mandible"}
PTVs = {"PTV56", "PTV63", "PTV70"}

mid, gt_oar, pr_oar = compute_avg_dvh_from_cache(df[df["ROI"].isin(OARs)])
plot_avg_dvh(mid, gt_oar, pr_oar, "DVH trung bình cho OAR", "dvh_avg_oar")

mid, gt_ptv, pr_ptv = compute_avg_dvh_from_cache(df[df["ROI"].isin(PTVs)])
plot_avg_dvh(mid, gt_ptv, pr_ptv, "DVH trung bình cho PTV", "dvh_avg_ptv", summary_df=summary, show_dxx_for_ptv=False)

print("Hoàn tất: Đã xuất bảng kết quả và biểu đồ DVH.")'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from provided_code.data_loader import DataLoader
from provided_code.dose_evaluation_class import DoseEvaluator

# ==== Dịch tên ROI sang tiếng Việt ====
ROI_VIETNAMESE = {
    "Brainstem": "Thân não",
    "SpinalCord": "Tủy sống",
    "RightParotid": "Tuyến mang tai phải",
    "LeftParotid": "Tuyến mang tai trái",
    "Larynx": "Thanh quản",
    "Esophagus": "Thực quản",
    "Mandible": "Xương hàm",
    "PTV56": "PTV56",
    "PTV63": "PTV63",
    "PTV70": "PTV70"
}

# ==== Thư mục dữ liệu ====
ref_dir = Path("provided-data/validation-pats")
pred_dir = Path("results/baseline/validation-predictions")

# ==== Tính các chỉ số cho từng bệnh nhân ====
patient_ids = [p.name for p in ref_dir.iterdir() if p.is_dir()]
results = []
results_cache = {}

for patient_id in patient_ids:
    try:
        ref_path = ref_dir / patient_id
        pred_path = pred_dir / f"{patient_id}.csv"
        ref_loader = DataLoader([ref_path])
        pred_loader = DataLoader([pred_path])
        evaluator = DoseEvaluator(ref_loader, pred_loader)
        evaluator.evaluate()

        dose_true = evaluator.reference_dose
        dose_pred = evaluator.prediction_dose.copy()
        roi_masks = {roi: evaluator.get_roi_mask(roi) for roi in ref_loader.full_roi_list}
        results_cache[patient_id] = (dose_true, dose_pred, roi_masks)

        for roi, mask in roi_masks.items():
            if mask is None or np.sum(mask) < 100:
                continue
            gt = dose_true[mask]
            pr = dose_pred[mask]
            results.append({
                "Patient_ID": patient_id,
                "ROI": roi,
                "D99_GT": np.percentile(gt, 1),
                "D60_GT": np.percentile(gt, 40),
                "D20_GT": np.percentile(gt, 80),
                "D2_GT": np.percentile(gt, 98),
                "Dmax_GT": np.max(gt),
                "Dmean_GT": np.mean(gt),
                "D99_Pred": np.percentile(pr, 1),
                "D60_Pred": np.percentile(pr, 40),
                "D20_Pred": np.percentile(pr, 80),
                "D2_Pred": np.percentile(pr, 98),
                "Dmax_Pred": np.max(pr),
                "Dmean_Pred": np.mean(pr),
            })
    except Exception as e:
        print(f"[!] Lỗi {patient_id}: {e}")

# ==== Tổng hợp bảng chỉ số tiếng Việt (GT ± SD, Pred ± SD, Sai số) ====
metrics = ["D99", "D60", "D20", "D2", "Dmax", "Dmean"]
df = pd.DataFrame(results)
formatted_rows = []

for roi in sorted(df["ROI"].unique()):
    roi_df = df[df["ROI"] == roi]
    if roi_df.empty: continue

    row = {"Cơ quan (ROI)": ROI_VIETNAMESE.get(roi, roi)}
    for metric in metrics:
        gt_col = f"{metric}_GT"
        pred_col = f"{metric}_Pred"
        if gt_col not in roi_df or pred_col not in roi_df:
            continue
        gt_vals = roi_df[gt_col]
        pred_vals = roi_df[pred_col]
        gt_mean, gt_std = gt_vals.mean(), gt_vals.std()
        pred_mean, pred_std = pred_vals.mean(), pred_vals.std()
        abs_err = np.mean(np.abs(gt_vals - pred_vals))

        row[f"{metric} (Thực tế ± SD)"] = f"{gt_mean:.2f} ± {gt_std:.2f}"
        row[f"{metric} (Dự đoán ± SD)"] = f"{pred_mean:.2f} ± {pred_std:.2f}"
        row[f"{metric} (Sai số tuyệt đối)"] = f"{abs_err:.2f}"
    formatted_rows.append(row)

formatted_df = pd.DataFrame(formatted_rows)
formatted_df.to_excel("dvh_metrics_error_detail.xlsx", index=False)
print("✅ Đã lưu: dvh_metrics_error_detail.xlsx")

# ==== Tạo bảng theo định dạng OpenKBP (GIỮ NGUYÊN) ====
def calculate_summary_table_openkbp(df: pd.DataFrame) -> pd.DataFrame:
    target_metrics = [
        ("PTV70", "D99"),
        ("PTV63", "D99"),
        ("PTV56", "D99"),
        ("Brainstem", "Dmax"),
        ("LeftParotid", "Dmean"),
        ("RightParotid", "Dmean"),
        ("SpinalCord", "Dmax"),
        ("Esophagus", "Dmean"),
        ("Larynx", "Dmean"),
        ("Mandible", "Dmax"),
    ]
    rows = []
    for roi, metric in target_metrics:
        gt_col = f"{metric}_GT"
        pr_col = f"{metric}_Pred"
        roi_df = df[df["ROI"] == roi]
        if gt_col not in roi_df or pr_col not in roi_df:
            continue
        gt_vals = roi_df[gt_col]
        pr_vals = roi_df[pr_col]
        diff_vals = pr_vals - gt_vals
        percent_diff = 100 * diff_vals / gt_vals.replace(0, np.nan)
        row = {
            "Chỉ số": f"{ROI_VIETNAMESE.get(roi, roi)} - {metric}",
            "Thực tế (Gy)": f"{gt_vals.mean():.2f}±{gt_vals.std():.2f}",
            "Dự đoán (Gy)": f"{pr_vals.mean():.2f}±{pr_vals.std():.2f}",
            "Hiệu số (Gy)": f"{diff_vals.mean():.2f}±{diff_vals.std():.2f} ({percent_diff.mean():.2f}%)"
        }
        rows.append(row)
    return pd.DataFrame(rows)

openkbp_table = calculate_summary_table_openkbp(df)
openkbp_table.to_excel("openkbp_style_summary.xlsx", index=False)
print("✅ Đã lưu: openkbp_style_summary.xlsx")

# ==== Tính DVH trung bình ====
def compute_avg_dvh_from_cache(df: pd.DataFrame, bins=100, max_dose=80):
    dose_bins = np.linspace(0, max_dose, bins)
    mid_bins = dose_bins[:-1]
    dvh_gt_dict, dvh_pred_dict = {}, {}
    for roi in df["ROI"].unique():
        roi_df = df[df["ROI"] == roi]
        gt_all, pred_all = [], []
        for _, row in roi_df.iterrows():
            pid = row["Patient_ID"]
            dose_true, dose_pred, roi_masks = results_cache[pid]
            mask = roi_masks.get(roi, None)
            if mask is None or np.sum(mask) == 0:
                continue
            gt = dose_true[mask]
            pr = dose_pred[mask]
            gt_hist, _ = np.histogram(gt, bins=dose_bins)
            pr_hist, _ = np.histogram(pr, bins=dose_bins)
            gt_cdf = np.cumsum(gt_hist[::-1])[::-1] / len(gt) * 100
            pr_cdf = np.cumsum(pr_hist[::-1])[::-1] / len(pr) * 100
            gt_all.append(gt_cdf)
            pred_all.append(pr_cdf)
        if gt_all: dvh_gt_dict[roi] = np.mean(gt_all, axis=0)
        if pred_all: dvh_pred_dict[roi] = np.mean(pred_all, axis=0)
    return mid_bins, dvh_gt_dict, dvh_pred_dict

# ==== Vẽ DVH ====
def plot_avg_dvh(mid_bins, dvh_gt, dvh_pred, title, save_path):
    plt.figure(figsize=(10, 6))
    for roi in dvh_gt:
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        name_vn = ROI_VIETNAMESE.get(roi, roi)
        plt.plot(mid_bins, dvh_gt[roi], label=f"{name_vn} - Thực tế", linestyle='-', color=color)
        if roi in dvh_pred:
            plt.plot(mid_bins, dvh_pred[roi], label=f"{name_vn} - Dự đoán", linestyle='--', color=color)
    plt.xlabel("Liều (Gy)")
    plt.ylabel("Tỷ lệ thể tích còn lại (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="medium")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path + ".pdf")
    plt.savefig(save_path + ".png", dpi=300)
    plt.close()

# ==== Vẽ DVH trung bình ====
OARs = {"Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Larynx", "Esophagus", "Mandible"}
PTVs = {"PTV56", "PTV63", "PTV70"}

mid, gt_oar, pr_oar = compute_avg_dvh_from_cache(df[df["ROI"].isin(OARs)])
plot_avg_dvh(mid, gt_oar, pr_oar, "DVH trung bình cho OAR", "dvh_avg_oar")

mid, gt_ptv, pr_ptv = compute_avg_dvh_from_cache(df[df["ROI"].isin(PTVs)])
plot_avg_dvh(mid, gt_ptv, pr_ptv, "DVH trung bình cho PTV", "dvh_avg_ptv")

print("✅ Hoàn tất: Đã xuất bảng kết quả và biểu đồ DVH.")
