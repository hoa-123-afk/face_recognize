import cv2
import os
from mtcnn import MTCNN
import uuid
import numpy as np
import imgaug.augmenters as iaa

# Khởi tạo MTCNN để phát hiện khuôn mặt
detector = MTCNN()

# Số lượng ảnh tối đa cần chụp
max_images = 100
count = 0

# Hàm augmentation để tăng số lượng ảnh
def augment_images(user_dir, num_augments=5):
    # Định nghĩa các phép augmentation
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Lật ngang với xác suất 50%
        iaa.Affine(rotate=(-20, 20)),  # Xoay ảnh từ -20 đến 20 độ
        iaa.Multiply((0.8, 1.2)),  # Thay đổi độ sáng
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Làm mờ Gaussian
    ])

    # Lấy danh sách ảnh trong thư mục
    image_files = [f for f in os.listdir(user_dir) if f.endswith('.jpg')]
    for image_file in image_files:
        image_path = os.path.join(user_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            # Tạo num_augments ảnh augment từ ảnh gốc
            for i in range(num_augments):
                augmented_image = seq(image=image)
                aug_id = str(uuid.uuid4())
                aug_path = os.path.join(user_dir, f"{aug_id}_aug.jpg")
                cv2.imwrite(aug_path, augmented_image)
                print(f"Đã lưu ảnh augment {aug_id} vào {aug_path}")

# Hàm chụp ảnh từ webcam, vẽ khung và lưu khuôn mặt
def capture_and_save_faces(name):
    global count
    # Tạo thư mục cho người dùng nếu chưa tồn tại
    user_dir = os.path.join('dataset', name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    # Mở webcam
    cap = cv2.VideoCapture(1)
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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Trích xuất khuôn mặt từ frame
            face_img = frame[y:y+h, x:x+w]
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

    # Áp dụng augmentation để tăng số lượng ảnh
    print("Bắt đầu augmentation...")
    augment_images(user_dir)
    print("Hoàn tất augmentation.")

# Nhập tên người dùng và chạy chương trình
name = input("Nhập tên của bạn: ")
capture_and_save_faces(name)