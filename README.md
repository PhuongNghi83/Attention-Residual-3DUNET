# Attention Residual 3D U-Net
Đây là một dự án nghiên cứu áp dụng mô hình Attention-Residual 3D U-Net để dự đoán phân bố liều xạ trị từ dữ liệu hình ảnh.

Attention-Residual-3DUNET/

  ├── main.py # Tập lệnh chính để chạy toàn bộ quy trình
  
  ├── model.py # Định nghĩa mô hình Attention-Residual 3D U-Net
  
  ├── requirements.txt # Các thư viện cần thiết
  
  ├── provided_code/ # Mã hỗ trợ (DataLoader, Evaluator, PredictionModel...)
  
  ├── provided-data/ # Dữ liệu mẫu hoặc hướng dẫn cách thêm dữ liệu
  
  ├── saved_model/ # Thư mục lưu model đã huấn luyện
  
  └── results/ # Kết quả đầu ra (DVH, heatmap, v.v.)
  
**Cách sử dụng**
   Bước 1: Chuẩn bị môi trường. Tạo môi trường ảo và cài đặt thư viện:
   
     python -m venv .venv

     pip install --upgrade pip
     
     pip install -r requirements.txt
     
   Bước 2: Chuẩn bị dữ liệu. Đặt dữ liệu vào thư mục provided-data/. Trong file main.py, kiểm tra lại các đường dẫn dữ liệu để chắc chắn rằng chương trình đọc đúng vị trí.
   
   Bước 3 : Chạy chương trình
   
     python main.py
     
**Quá trình này sẽ:**

    Đọc dữ liệu từ provided-data/
    
    Xây dựng và huấn luyện mô hình Attention-Residual 3D U-Net
    
    Lưu mô hình vào saved_model/
    
    Xuất kết quả đánh giá (VD: DVH, heatmap) trong thư mục results/
    
**Yêu cầu hệ thống:**

   Python 3.8 trở lên

   TensorFlow 2.x (khuyến nghị dùng tensorflow==2.13.0)
   
   Các thư viện khác được liệt kê trong requirements.txt
   


# Attention Residual 3D U-Net
This is a research project applying the Attention-Residual 3D U-Net model to predict radiation dose distribution from medical imaging data.

Attention-Residual-3DUNET/

  ├── main.py           # Main script to run the entire workflow
  
  ├── model.py          # Definition of the Attention-Residual 3D U-Net model
  
  ├── requirements.txt  # Required libraries
  
  ├── provided_code/    # Supporting code (DataLoader, Evaluator, PredictionModel...)
  
  ├── provided-data/    # Sample data or instructions for adding new data
  
  ├── saved_model/      # Directory to save trained models
  
  └── results/          # Output results (DVH, heatmap, etc.)
  
**How to Use**

  Step 1: Set up the environment. Ceate a virtual environment and install dependencies:
  
    python -m venv .venv

    pip install --upgrade pip
    
    pip install -r requirements.txt
    
  Step 2: Prepare the data. Place your data in the provided-data/ folder. In main.py, check the data paths to ensure the program reads the correct location.
  
  Step 3: Run the program
  
    python main.py
    
**This process will:**

    Load data from provided-data/
    
    Build and train the Attention-Residual 3D U-Net model.
    
    Save the trained model in saved_model/
    
    Generate evaluation results (e.g., DVH, heatmaps) in the results/ directory
    
**System Requirements:** 

  Python 3.8 or higher 
  
  TensorFlow 2.x (recommended: tensorflow==2.13.0)
  
  Other dependencies listed in requirements.txt
  
