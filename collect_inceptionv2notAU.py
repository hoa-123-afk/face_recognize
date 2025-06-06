import cv2
import os
from mtcnn import MTCNN
import uuid

# Khởi tạo MTCNN để phát hiện khuôn mặt
detector = MTCNN()

# Số lượng ảnh tối đa cần chụp
max_images = 20
count = 0


# Hàm chụp ảnh từ webcam, vẽ khung và lưu khuôn mặt
def capture_and_save_faces(name):
    global count
    # Tạo thư mục cho người dùng nếu chưa tồn tại
    user_dir = os.path.join('datasets', name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    # Mở webcam
    cap = cv2.VideoCapture(0)  # Changed to 0 (default webcam); use 1 if needed
    if not cap.isOpened():
        print("Không thể mở webcam")
        return

    while count < max_images:
        # Đọc frame từ webcam
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ webcam")
            break

        # Phát hiện khuôn mặt trong frame
        faces = detector.detect_faces(frame)
        for face in faces:
            if count >= max_images:
                break
            # Lấy tọa độ và kích thước của khuôn mặt
            x, y, w, h = face['box']
            # Vẽ khung màu xanh xung quanh khuôn mặt
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Trích xuất khuôn mặt từ frame
            face_img = frame[y:y + h, x:x + w]
            # Tạo ID duy nhất cho khuôn mặt
            face_id = str(uuid.uuid4())
            # Lưu khuôn mặt vào thư mục của người dùng
            face_path = os.path.join(user_dir, f"{face_id}.jpg")
            cv2.imwrite(face_path, face_img)
            print(f"Đã lưu khuôn mặt {face_id} vào {face_path}")
            count += 1
            print(f"Số ảnh đã chụp: {count}/{max_images}")

        # Hiển thị frame trên cửa sổ
        cv2.imshow('Webcam', frame)
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Thông báo khi đã chụp đủ 100 ảnh
    if count >= max_images:
        print(f"Đã chụp đủ {max_images} ảnh. Chương trình sẽ dừng.")

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()


# Nhập tên người dùng và chạy chương trình
name = input("Nhập tên của bạn: ")
capture_and_save_faces(name)