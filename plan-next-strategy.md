1. Tăng cường bộ dữ liệu testing (hiện tại chỉ có 64 clips với 4 unsafe-class)
    - Sử dụng các phương pháp tăng cường bộ dữ liệu testing - Data augmentation
    - Sử dụng testing xoay vòng.
2. Thử dùng các dòng model YOLO26 để thực nghiệm đánh giá.
3. Kiến trúc - model
    - Dựng lại kiến trúc mô hình với YOLO26 và training lại
    - Sử dụng backbone hiệu quả
    - Lưu lại kết quả của YOLO layer đó để đánh giá
    - Record log, và dùng thư viện để lưu lại tất cả kết quả thực nghiệm (CometML)
    - Nghiên cứu sử dụng Transformer hiệu quả (Attention), Thử phần Transformer với các mô hình học sâu khác
4. Đánh bbox lại nhiều frame hơn, tập trung hơn, chính xác hơn.
    - Tracking tất cả các đối tượng khi Tranformer
    - Gắn id nhận diện đối tượng (YOLOv26)
4. Các chỉ số đánh giá
    - Sử dụng các công thức đánh giá của paper
    - Tốc độ xử lý (s, ms / frame, clip) trên Jetson
    - Sử dụng nhiều độ đo, nhiều công thức trong xử lý ảnh và video, realtime


    Các hội nghị của Computer Vision - CVF The computer vision foundation
    Các tạp chí của Computer Vision - Xem họ đang làm gì, sử dụng metric nào, có cải tiến gì mới 
    https://openaccess.thecvf.com/CVPR2025?day=2025-06-13