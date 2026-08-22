# ============================================================
# FOOTBALL TACTICAL VISION — MODULE 4: EDGE ONNX OPTIMIZATION
# Author: Truong Thuan An
# ============================================================
import os
import time
import numpy as np
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_and_quantize(pt_model_path: str, output_name: str):
    """
    1. Chuyển PyTorch (.pt) sang ONNX Static Graph
    2. Nén INT8 để giảm 4 lần dung lượng
    """
    print(f"\n🚀 Đang xử lý mô hình: {pt_model_path}")
    
    # 1. Export ONNX (FP32)
    onnx_fp32 = f"{output_name}_fp32.onnx"
    model = YOLO(pt_model_path)
    model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    
    # Đổi tên file mặc định của YOLO thành tên ta muốn
    default_export_name = pt_model_path.replace(".pt", ".onnx")
    if os.path.exists(default_export_name):
        if os.path.exists(onnx_fp32): os.remove(onnx_fp32)
        os.rename(default_export_name, onnx_fp32)
        
    # 2. Nén INT8
    onnx_int8 = f"{output_name}_int8.onnx"
    print(f"⚡ Đang nén INT8 cho {output_name}...")
    quantize_dynamic(
        model_input=onnx_fp32,
        model_output=onnx_int8,
        weight_type=QuantType.QInt8
    )
    
    # 3. Tính toán dung lượng
    size_pt = os.path.getsize(pt_model_path) / (1024*1024)
    size_fp32 = os.path.getsize(onnx_fp32) / (1024*1024)
    size_int8 = os.path.getsize(onnx_int8) / (1024*1024)
    
    print(f"✅ HOÀN THÀNH {output_name}:")
    print(f"   • Kích thước PyTorch gốc: {size_pt:.1f} MB")
    print(f"   • Kích thước ONNX FP32:   {size_fp32:.1f} MB")
    print(f"   • Kích thước ONNX INT8:   {size_int8:.1f} MB (Giảm {(1 - size_int8/size_fp32)*100:.1f}%)")
    
    return onnx_int8

if __name__ == "__main__":
    print("="*60)
    print("⚽ BẮT ĐẦU TỐI ƯU HÓA MÔ HÌNH KLTN CỦA TRƯƠNG THUẬN AN")
    print("="*60)
    
    # Tối ưu hóa mô hình Player (best_player.pt)
    if os.path.exists("best_player.pt"):
        export_and_quantize("best_player.pt", "player_detector")
    else:
        print("⚠️ Không tìm thấy best_player.pt")
        
    # Tối ưu hóa mô hình Pitch Keypoint (best_pitch.pt)
    if os.path.exists("best_pitch.pt"):
        export_and_quantize("best_pitch.pt", "pitch_keypoint")
    else:
        print("⚠️ Không tìm thấy best_pitch.pt")
