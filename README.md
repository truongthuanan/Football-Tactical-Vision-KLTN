# ⚽ Phân tích Chiến thuật Bóng đá qua Thị giác Máy tính (Tactical Vision)

**Khóa luận Tốt nghiệp 2026**  
**Sinh viên thực hiện:** Trương Thuận An  
**Giáo viên hướng dẫn:** TS. Nguyễn Đăng Bình  
**Đơn vị:** Khoa Công nghệ Thông tin – Trường Đại học Khoa học, Đại học Huế  

---

## 📖 Giới thiệu Dự án
Dự án này là hệ thống thị giác máy tính tự động phân tích video bóng đá, được thiết kế để hoạt động hoàn toàn ngoại tuyến (offline) trên Google Colab. Hệ thống kế thừa kiến trúc từ Roboflow/sports, mở rộng từ 6 lên **9 Module hoàn chỉnh**, bao gồm việc trích xuất các thông số chiến thuật nâng cao như: Bản đồ nhiệt, Biểu đồ Voronoi kiểm soát không gian, và Quỹ đạo bóng.

Đặc biệt, hệ thống đã được đánh giá định lượng chéo miền trên 3 cấp độ video: **Ngoại hạng Anh (EPL 4K)**, **Shopee Cup (FHD)** và **V-League (720p)** nhằm kiểm chứng khả năng ứng dụng thực tiễn cho bóng đá Việt Nam.

## ⚙️ Kiến trúc Đường ống 9 Module
Hệ thống xử lý tuần tự qua 9 bước (Modules) cốt lõi:
1. **Module 1 (Phát hiện đối tượng):** Sử dụng YOLOv8x để phát hiện Cầu thủ, Thủ môn, Trọng tài, Bóng.
2. **Module 2 (Theo dõi đối tượng):** Thuật toán ByteTrack giúp duy trì định danh (ID) liên tục.
3. **Module 3 (Phân loại đội bóng):** Trích xuất đặc trưng hình ảnh bằng SigLIP, giảm chiều UMAP và gom cụm KMeans (Không cần khai báo màu áo trước).
4. **Module 4 (Điểm mốc sân):** Dùng YOLOv8x-Pose phát hiện 32 điểm mốc chuẩn của sân bóng.
5. **Module 5 (Phép chiếu Homography):** Chuyển đổi tọa độ 2D từ video sang tọa độ thực trên bề mặt sân phẳng.
6. **Module 6 (Minimap & Vệt theo dõi):** Trực quan hóa bản đồ nhỏ 2D và vệt di chuyển của từng cầu thủ.
7. **Module 7 (Voronoi):** Phân vùng diện tích kiểm soát không gian của mỗi đội. *(Đóng góp mới)*
8. **Module 8 (Bản đồ nhiệt - Heatmap):** Tích lũy vị trí để tạo bản đồ mật độ hoạt động. *(Đóng góp mới)*
9. **Module 9 (Quỹ đạo bóng):** Theo vết và vẽ đường bay của quả bóng. *(Đóng góp mới)*

## 📊 Kết quả Thực nghiệm Đáng chú ý
- **Phát hiện bóng (Ball Detection):** 84,2% (EPL) | 43,9% (V-League).
- **Phân loại đội bóng:** Độ chính xác đạt 91,2% (EPL) | 78,3% (V-League).
- **Lỗi phép chiếu Homography:** Tỷ lệ lỗi rất thấp ở EPL (0,8%) nhưng tăng lên 8,5% ở V-League do thiếu điểm mốc.
- **Tốc độ xử lý:** ~18,7 FPS (Xử lý Offline trên GPU NVIDIA L4).

## 🚀 Hướng dẫn Sử dụng (Google Colab)
Toàn bộ mã nguồn đã được đóng gói thành một file Jupyter Notebook duy nhất để dễ dàng chạy thử nghiệm.
1. Tải file `1_Main_Pipeline_Inference.ipynb` lên Google Colab.
2. Thiết lập môi trường chạy với phần cứng **GPU (T4 hoặc L4)**.
3. Thêm cấu hình **Secrets (🔑)** trong Colab:
   - `HF_TOKEN`: Token của HuggingFace.
   - `ROBOFLOW_API_KEY`: API Key của tài khoản Roboflow của bạn.
4. Chạy tuần tự từ trên xuống dưới (Run All). Kết quả đầu ra sẽ là một video `.mp4` hoàn chỉnh đã được vẽ đè các lớp phân tích chiến thuật.

## 🎓 Lời cảm ơn
Xin gửi lời cảm ơn sâu sắc đến TS. Nguyễn Đăng Bình đã tận tình hướng dẫn, định hướng khoa học và hỗ trợ tác giả hoàn thành khóa luận này. Đồng thời, xin cảm ơn cộng đồng mã nguồn mở (Roboflow, Ultralytics) đã cung cấp các công cụ nền tảng vô giá.
