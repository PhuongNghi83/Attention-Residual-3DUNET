import shutil
from pathlib import Path

from provided_code import DataLoader, DoseEvaluator, PredictionModel, get_paths

if __name__ == "__main__":

    prediction_name = "baseline"  # Tên mô hình (thư mục lưu kết quả)
    test_time = False  # Chạy trên validation set (False) hay test set (True)
    num_epochs = 30  # Số epoch huấn luyện (nên tăng lên khi chạy chính thức)

    # Thiết lập đường dẫn các thư mục dữ liệu và lưu kết quả
    primary_directory = Path().resolve()
    provided_data_dir = primary_directory / "provided-data"
    training_data_dir = provided_data_dir / "train-pats"
    validation_data_dir = provided_data_dir / "validation-pats"
    testing_data_dir = provided_data_dir / "test-pats"
    results_dir = primary_directory / "results"

    # Tải danh sách thư mục chứa dữ liệu huấn luyện
    training_plan_paths = get_paths(training_data_dir)

    #  Khởi tạo mô hình và huấn luyện
    data_loader_train = DataLoader(training_plan_paths)
    dose_prediction_model_train = PredictionModel(data_loader_train, results_dir, prediction_name, "train")
    dose_prediction_model_train.train_model(num_epochs, save_frequency=1, keep_model_history=20)

    #  Chuẩn bị tập validation hoặc test để đánh giá
    hold_out_data_dir = validation_data_dir if not test_time else testing_data_dir
    stage_name, _ = hold_out_data_dir.stem.split("-")  # "validation" hoặc "test"
    hold_out_plan_paths = get_paths(hold_out_data_dir)

    #  Dự đoán liều cho tập hold-out
    data_loader_hold_out = DataLoader(hold_out_plan_paths)
    dose_prediction_model_hold_out = PredictionModel(data_loader_hold_out, results_dir, model_name=prediction_name, stage=stage_name)
    dose_prediction_model_hold_out.predict_dose(epoch=num_epochs)

    #  Đánh giá mô hình với DVH score và Dose score
    data_loader_hold_out_eval = DataLoader(hold_out_plan_paths)
    prediction_paths = get_paths(dose_prediction_model_hold_out.prediction_dir, extension="csv")
    hold_out_prediction_loader = DataLoader(prediction_paths)
    dose_evaluator = DoseEvaluator(data_loader_hold_out_eval, hold_out_prediction_loader)

    # In kết quả nếu có dữ liệu đánh giá
    if not data_loader_hold_out_eval.patient_paths:
        print(" Không có thông tin bệnh nhân để tính toán đánh giá.")
    else:
        dose_evaluator.evaluate()
        dose_score, dvh_score = dose_evaluator.get_scores()
        print(f"\n Kết quả đánh giá trên tập {stage_name}:")
        print(f"\t• DVH score (phân bố thể tích): {dvh_score:.3f}")
        print(f"\t• Dose score (liều sai lệch): {dose_score:.3f}")

    # Nén các file dự đoán để nộp (submissions)
    submission_dir = results_dir / "submissions"
    submission_dir.mkdir(exist_ok=True)
    shutil.make_archive(str(submission_dir / prediction_name), "zip", dose_prediction_model_hold_out.prediction_dir)
    print(f"Đã nén kết quả dự đoán vào: {submission_dir / (prediction_name + '.zip')}")