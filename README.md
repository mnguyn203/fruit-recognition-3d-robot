# Hệ thống Thị giác cho Robot Tay Máy Gắp Vật Thể trong Môi Trường Ánh Sáng Thay Đổi

## Giới thiệu
Dự án này xây dựng một **hệ thống thị giác máy tính (Computer Vision System)** phục vụ cho **robot tay máy gắp vật thể** trong môi trường có **điều kiện ánh sáng thay đổi** như ánh sáng yếu, ánh sáng không đồng đều và bóng đổ.

Hệ thống kết hợp **YOLOv8**, **camera RGB-D Intel RealSense** và các kỹ thuật xử lý ảnh thích nghi ánh sáng nhằm cung cấp **tọa độ 3D ổn định và chính xác** của vật thể, hỗ trợ robot trong quá trình định vị và gắp vật.

---

## Mục tiêu
- Phát hiện và phân đoạn vật thể bằng YOLOv8
- Tính toán vị trí **3D (x, y, z)** của vật thể từ dữ liệu RGB-D
- Giảm nhiễu do bóng đổ và điều kiện ánh sáng kém
- Theo dõi vật thể ổn định theo thời gian
- Cung cấp dữ liệu đầu vào cho robot tay máy gắp vật thể

---

## Kiến trúc hệ thống

```
RGB Image ──┐
├─> Image Enhancement (CLAHE + Gamma)
Depth Image ─┘
↓
YOLOv8 Segmentation
↓
Shadow Filtering (Depth + Brightness)
↓
3D Reconstruction (RGB-D)
↓
Object Tracking & Smoothing
↓
Output: Object 3D Position
```


---

## Công nghệ & Thư viện sử dụng
- **YOLOv8 (Ultralytics)** – Object Detection & Segmentation
- **PyTorch** – Deep Learning framework
- **OpenCV** – Xử lý ảnh
- **Intel RealSense SDK (pyrealsense2)** – Camera RGB-D
- **NumPy** – Tính toán số
- **CUDA** – Tăng tốc GPU (nếu khả dụng)

---

## Phần cứng
- Camera: **Intel RealSense (RGB-D)**
- Robot: Tay máy gắp vật thể (chưa tích hợp trực tiếp trong phiên bản hiện tại)

---

## Các thành phần chính

### VisionProcessor
- Thu nhận ảnh RGB và Depth từ camera
- Cân bằng ánh sáng bằng **CLAHE** và **Gamma Correction**
- Chạy YOLOv8 segmentation
- Tính toán tọa độ 3D từ dữ liệu depth
- Lọc nhiễu và bóng đổ dựa trên:
  - Số lượng điểm depth hợp lệ
  - Độ lệch chuẩn chiều sâu
  - Độ sáng vùng vật thể

---

### ObjectTracker
- Theo dõi vật thể theo từng frame
- Gán ID cố định theo class
- Làm mượt vị trí 3D bằng **Exponential Moving Average**
- Loại bỏ vật thể bị mất trong nhiều frame liên tiếp
- Giảm hiện tượng bounding box và tọa độ bị nhảy

---

### Cấu trúc dữ liệu Object3D
Mỗi vật thể được biểu diễn bởi:
- `name`: Tên vật thể
- `center_3d`: Tọa độ 3D (x, y, z)
- `size_3d`: Kích thước 3D
- `confidence`: Độ tin cậy
- `distance`: Khoảng cách tới camera
- `mask`: Mask segmentation

---

## Kết quả & Đánh giá

### Độ ổn định
- Bounding box và mask ổn định theo thời gian
- Tọa độ 3D ít dao động nhờ cơ chế smoothing

### Khả năng thích nghi ánh sáng
- Hoạt động tốt trong điều kiện:
  - Ánh sáng yếu
  - Ánh sáng không đồng đều
  - Có bóng đổ

### Phù hợp cho robot gắp vật
- Dữ liệu 3D đủ chính xác cho bài toán định vị
- Có thể tích hợp vào pipeline điều khiển robot

## Hạn chế
- Phụ thuộc vào chất lượng dữ liệu depth từ camera
- Hiệu năng giảm khi vật thể ở khoảng cách xa
- Chưa tích hợp trực tiếp điều khiển robot tay máy

---

## Hướng phát triển
- Tích hợp điều khiển robot tay máy (ROS / MoveIt)
- Ước lượng pose 6D của vật thể
- Huấn luyện mô hình với dataset thực tế hơn
- Tối ưu tốc độ xử lý real-time
- Ứng dụng trong môi trường công nghiệp

---

## Ứng dụng
- Robot gắp và phân loại vật thể
- Kho vận thông minh
- Sản xuất tự động
- Robot dịch vụ

---

## 📜 Kết luận
Dự án cho thấy khả năng xây dựng một **hệ thống thị giác hiệu quả cho robot tay máy** trong môi trường ánh sáng phức tạp, kết hợp giữa **Deep Learning, RGB-D và xử lý ảnh truyền thống**.

> ⚠️ Dự án phục vụ mục đích học tập và nghiên cứu, chưa thay thế hệ thống robot công nghiệp hoàn chỉnh.
