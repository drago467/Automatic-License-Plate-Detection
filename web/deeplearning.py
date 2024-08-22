import os
import subprocess
import sys
import cv2
import numpy as np
# import matplotlib as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from util import get_car, read_license_plate, write_csv


# Clone the YOLOv10 repository (chỉ thực hiện nếu bạn chưa có thư mục 'yolov10')
# if not os.path.exists('yolov10'):
#     subprocess.check_call(['git', 'clone', 'https://github.com/THU-MIG/yolov10.git'])

# Navigate to the yolov10 directory
# os.chdir('yolov10')

# Install the requirements
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', '-q', '-r', 'requirements.txt'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'])

from ultralytics import YOLOv10

# Tải mô hình nếu cần (bỏ qua nếu bạn đã có best.pt)
# url = 'http://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10n.pt'
# filename = 'yolov10n.pt'
# urllib.request.urlretrieve(url, filename)

# Đảm bảo rằng tệp 'best.pt' đã tồn tại

TRAINED_MODEL_PATH = 'best.pt'
if not os.path.exists(TRAINED_MODEL_PATH):
    raise FileNotFoundError(f"Tệp mô hình '{TRAINED_MODEL_PATH}' không tồn tại.")

# Khởi tạo mô hình YOLOv10
model = YOLOv10(TRAINED_MODEL_PATH)

def predict_and_extract(model, image_path, filename, conf_threshold=0.5):
    results = model.predict(source=image_path, imgsz=1800, conf=conf_threshold)

    # Lưu ảnh kết quả
    annotated_filename = f'{filename}_annotated.jpg'
    annotated_filepath = os.path.join('./static/predict/', annotated_filename)
    annotated_img = results[0].plot()
    cv2.imwrite(annotated_filepath, annotated_img)

    # Lấy các bounding box và tên đối tượng
    boxes = results[0].boxes  # Các hộp dự đoán
    names = results[0].names  # Tên các lớp dự đoán

    coords = []  # Danh sách lưu trữ tọa độ [x1, y1, x2, y2] và tên đối tượng

    for j in range(len(boxes)):
        x1, y1, x2, y2 = boxes[j].xyxy[0]
        label = names[int(boxes[j].cls[0])]
        coords.append([label, int(x1), int(y1), int(x2), int(y2)])

    return coords

def OCR(model, path, filename):
    img = np.array(load_img(path))
    coords = predict_and_extract(model, path, filename)
    text = []
    for idx, coord in enumerate(coords):
        label, x1, y1, x2, y2 = coord
        cropped_img = img[y1:y2, x1:x2]

        cropped_filename = f'{filename}_{idx}.jpg'
        cropped_filepath = os.path.join('./static/roi/', cropped_filename)
        cv2.imwrite(cropped_filepath, cropped_img)

        license_plate_text, license_plate_text_score = read_license_plate(cropped_filepath)
        text.append([cropped_filename, license_plate_text, license_plate_text_score])

    return text

# Ví dụ sử dụng hàm:
image_path = 'test06.jpg'
print(OCR(model, image_path, 'test'))