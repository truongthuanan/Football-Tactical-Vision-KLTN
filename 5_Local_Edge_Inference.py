# ============================================================
# FOOTBALL TACTICAL VISION — MODULE 5: LOCAL EDGE INFERENCE
# Author: Truong Thuan An
# ============================================================
import os
import cv2
import time
from ultralytics import YOLO

# 1. CẤU HÌNH ĐẦU VÀO / ĐẦU RA
INPUT_FILE = "video.mp4"
OUTPUT_FILE = "output_result.mp4"

def run_edge_inference():
    print("="*60)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG LOCAL EDGE INFERENCE (ONNX INT8)")
    print("="*60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ LỖI: Không tìm thấy file {INPUT_FILE}.")
        print("Vui lòng copy 1 ảnh/video vào thư mục này và đổi tên biến INPUT_FILE trong code.")
        return

    # 2. LOAD MÔ HÌNH ONNX SIÊU NHẸ (Chạy trên CPU)
    print("\n⏳ Đang load mô hình ONNX từ ổ cứng...")
    try:
        player_model = YOLO("player_detector_fp32.onnx")
        print("✅ Load thành công Player Detector FP32!")
    except Exception as e:
        print(f"❌ Lỗi load mô hình Player: {e}")
        return

    # 3. CHẠY SUY LUẬN (INFERENCE)
    print(f"\n🔍 Đang phân tích file: {INPUT_FILE}...")
    start_time = time.time()
    
    # Hàm predict tự động load video, dự đoán từng frame và tự động ghép thành video mới nếu save=True
    player_model.predict(
        source=INPUT_FILE, 
        conf=0.3,       # Ngưỡng tin cậy 30%
        device="cpu",   # Ép chạy trên CPU
        save=True,      # Bật tính năng tự động vẽ và lưu kết quả của YOLO
        verbose=False
    )
    
    inference_time = time.time() - start_time
    print(f"⚡ Xử lý xong toàn bộ video trong {inference_time:.2f} giây!")
    
    # YOLO tự động lưu file kết quả vào thư mục runs/detect/predict
    print("="*60)
    print("📸 YOLO đã tự động lưu file kết quả (đuôi .avi) vào thư mục: runs/detect/predict")
    print("Mời bạn vào thư mục đó mở file video lên xem thành quả nhé!")

if __name__ == "__main__":
    run_edge_inference()
