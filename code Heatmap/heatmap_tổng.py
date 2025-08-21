import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from provided_code.data_loader import DataLoader
from provided_code.dose_evaluation_class import DoseEvaluator

# ==== Hàm trích xuất 3 lát cắt chính (axial, sagittal, coronal) ====
def get_slices(volume):
    z, y, x = 64, 64, 64
    return volume[z, :, :], volume[:, y, :], volume[:, :, x]

# ==== Vẽ và TRẢ VỀ ảnh PIL thay vì hiển thị ngay ====
def generate_patient_heatmap(ct_image, dose_pred, dose_true, patient_id):
    diff = dose_pred - dose_true

    ct_ax, ct_sag, ct_cor = get_slices(ct_image)
    pred_ax, pred_sag, pred_cor = get_slices(dose_pred)
    gt_ax, gt_sag, gt_cor = get_slices(dose_true)
    df_ax, df_sag, df_cor = get_slices(diff)

    fig, axs = plt.subplots(3, 4, figsize=(8, 7))
    titles = ["Ảnh CT", "Liều dự đoán", "Liều thực tế", "Chênh lệch"]
    views = ["Lát ngang (Axial)", "Lát dọc (Sagittal)", "Lát trước-sau (Coronal)"]

    cmap_ct = "gray"
    cmap_dose = "jet"
    cmap_diff = "coolwarm"

    for row, (ct, pr, gt, df) in enumerate([(ct_ax, pred_ax, gt_ax, df_ax),
                                            (ct_sag, pred_sag, gt_sag, df_sag),
                                            (ct_cor, pred_cor, gt_cor, df_cor)]):
        axs[row, 0].imshow(ct, cmap=cmap_ct)
        axs[row, 1].imshow(pr, cmap=cmap_dose, vmin=0, vmax=70)
        axs[row, 2].imshow(gt, cmap=cmap_dose, vmin=0, vmax=70)
        axs[row, 3].imshow(df, cmap=cmap_diff, vmin=-20, vmax=20)
        for col in range(4):
            axs[row, col].axis("off")
            if row == 0:
                axs[row, col].set_title(titles[col], fontsize=10, pad=10)
        axs[row, 0].set_ylabel(views[row], fontsize=10, labelpad=12)

    # Thanh màu
    cbar_ax_dose = fig.add_axes([0.15, 0.06, 0.7, 0.015])
    im_dose = axs[0, 1].imshow(pred_ax, cmap=cmap_dose, vmin=0, vmax=70)
    cbar = plt.colorbar(im_dose, cax=cbar_ax_dose, orientation='horizontal')
    cbar.set_label("Đơn vị: Gy", fontsize=10)

    cbar_ax_diff = fig.add_axes([0.91, 0.25, 0.015, 0.5])
    im_diff = axs[0, 3].imshow(df_ax, cmap=cmap_diff, vmin=-20, vmax=20)
    cbar_diff = plt.colorbar(im_diff, cax=cbar_ax_diff)
    cbar_diff.set_label("Chênh lệch liều (Gy)", fontsize=10)

    fig.text(0.5, 1, f"BẢN ĐỒ PHÂN BỐ LIỀU BỨC XẠ bệnh nhân: {patient_id}" , ha='center', va='top',
             fontsize=12, fontweight='bold')
    plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.12, wspace=0.05, hspace=0.12)

    # Chuyển sang ảnh PIL để ghép sau
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# ==== GHÉP 5 ẢNH THEO CHIỀU DỌC ====
def combine_patient_plots_5x1(images):
    w, h = images[0].size
    total_h = h * 5
    canvas = Image.new("RGB", (w, total_h), color="white")
    for i, img in enumerate(images):
        canvas.paste(img, (0, i * h))
    return canvas

# ==== CHƯƠNG TRÌNH CHÍNH ====
if __name__ == "__main__":
    ref_dir = Path("provided-data/validation-pats")
    pred_dir = Path("results/baseline/validation-predictions")

    all_patient_ids = [f"pt{i}" for i in range(201, 241)]
    group_size = 5
    all_images = []

    for patient_id in all_patient_ids:
        try:
            ref_path = ref_dir / patient_id
            pred_path = pred_dir / f"{patient_id}.csv"
            if not pred_path.exists():
                print(f"❌ Thiếu file dự đoán cho {patient_id}")
                continue

            ref_loader = DataLoader([ref_path])
            ref_loader.set_mode("training_model")
            pred_loader = DataLoader([pred_path])

            evaluator = DoseEvaluator(reference_data_loader=ref_loader, prediction_loader=pred_loader)
            reference_batch = ref_loader.get_patients([patient_id])
            evaluator.reference_batch = reference_batch
            evaluator.evaluate()

            ct_image = np.squeeze(reference_batch.ct[0])
            dose_true = evaluator.reference_dose.reshape(128, 128, 128)
            dose_pred = evaluator.prediction_dose.reshape(128, 128, 128)

            img = generate_patient_heatmap(ct_image, dose_pred, dose_true, patient_id)
            all_images.append(img)

        except Exception as e:
            print(f"❌ Lỗi với {patient_id}: {e}")

    # === Ghép từng nhóm 5 ảnh và hiển thị ===
    for i in range(0, len(all_images), group_size):
        group = all_images[i:i + group_size]
        if len(group) == 5:
            combined = combine_patient_plots_5x1(group)
            combined.show()
        else:
            print(f"⚠️ Nhóm {i//5 + 1} có ít hơn 5 ảnh — bỏ qua")
