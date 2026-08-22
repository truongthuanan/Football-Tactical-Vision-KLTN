
# ⚽ Football Tactical Vision (KLTN)

> **Thesis:** Application of YOLOv8 and ByteTrack in Football Tactical Analysis  
> **Author:** Truong Thuan An | **Institution:** Hue University of Sciences

An automated, end-to-end computer vision pipeline for football tactical analysis. This system processes broadcast football videos to detect players, classify teams, track movements, and project the real-world action onto a 2D tactical minimap in near real-time.

<img width="800" height="492" alt="demo" src="https://github.com/user-attachments/assets/3925f559-7b7d-4067-91b7-0696f28d2332" />
---

## ✨ Key Features

*   **Robust Object Detection & Tracking:** Detects players, referees, and the ball using fine-tuned **YOLOv8x**, while maintaining consistent IDs across frames using **ByteTrack**.
*   **Unsupervised Team Classification:** Automatically clusters and classifies teams based on jersey colors using **SigLIP** and **UMAP**—no manual color input required!
*   **Spatial Projection (Homography):** Detects 32 pitch keypoints to compute a Homography matrix (DLT + RANSAC), mapping camera view coordinates to a 2D top-down minimap.
*   **Advanced Tactical Visualizations:**
    *   🗺️ **2D Minimap:** Real-time player positioning.
    *   🕸️ **Voronoi Diagrams:** Analyzes space control and pressing zones.
    *   🔥 **KDE Heatmaps:** Tracks player density and hotspots over time.
    *   ⚽ **Ball Trajectory:** Interpolates and visualizes ball movement.
*   **Edge Deployment (New):** Exported trained PyTorch models to **ONNX** format and applied **INT8 Quantization** to enable lightweight and high-speed local CPU inference.

## 🛠️ Tech Stack

*   **Deep Learning:** PyTorch, Ultralytics (YOLOv8, YOLOv8-Pose)
*   **Computer Vision:** OpenCV (Homography mapping, Perspective Transform)
*   **Tracking & Clustering:** ByteTrack, SigLIP, UMAP, scikit-learn
*   **Data Processing & Visualization:** NumPy, Matplotlib, SciPy
*   **Deployment & Optimization:** ONNX Runtime, INT8 Quantization

## 📊 Performance & Results

The system was rigorously evaluated across multi-domain datasets (English Premier League, Shopee Cup, V-League) to analyze the impact of *Domain Shift* and *Domino Errors* in low-angle broadcasts.

*   **Player & Ball Detection:** `mAP50 = 88.0%` (Trained on custom annotated dataset)
*   **Pitch Keypoint Detection:** `mAP50-Pose = 99.5%` (32 keypoints)
*   **Processing Speed:** `18.7 FPS` (Evaluated on NVIDIA L4 GPU)

## 📂 Repository Structure

```text
├── 1_Main_Pipeline_Inference.ipynb       # Full 9-module inference pipeline (Run this for demo)
├── 2_Train_Player_Detector_YOLOv8...     # Training notebook for Object Detection
├── 3_Train_Pitch_Keypoint_Detector...    # Training notebook for Keypoint Detection
├── 4_Edge_ONNX_Optimization.py           # Script to export PyTorch models to ONNX and apply INT8 Quantization
├── 5_Local_Edge_Inference.py             # Script to run fast standalone inference on local CPU
└── README.md
```
