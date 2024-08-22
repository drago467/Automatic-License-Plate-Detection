import os
import subprocess
import sys
import cv2
import numpy as np
import pytesseract as pt
# import matplotlib as plt
# from tensorflow.keras.preprocessing.image import load_img, img_to_array


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

# # Đường dẫn đến hình ảnh cần dự đoán
# IMAGE_URL = 'car.jpg'  # Đảm bảo rằng hình ảnh này có sẵn trong thư mục làm việc của bạn
#
# # Thiết lập ngưỡng confidence
# CONF_THRESHOLD = 0.3
#
# # Chạy mô hình trên hình ảnh
# results = model.predict(source=IMAGE_URL,
#                         imgsz=1800,
#                         conf=CONF_THRESHOLD)
#
# # Annotate the image with results
# annotated_img = results[0].plot()
#
# # Hiển thị hình ảnh kết quả sử dụng OpenCV
# cv2.imshow("Annotated Image", annotated_img)
# cv2.waitKey(0)  # Đợi một phím bấm để đóng cửa sổ
# cv2.destroyAllWindows()
#
# # Tùy chọn: In kết quả để phân tích thêm
# print(results)

def predict_and_extract(model, image_path, conf_threshold=0.3):
    # Thực hiện dự đoán
    results = model.predict(source=image_path, imgsz=1800, conf=conf_threshold)

    # Lấy ảnh đã dự đoán
    annotated_img = results[0].plot()

    # Lấy các bounding box và tên đối tượng
    boxes = results[0].boxes  # Các hộp dự đoán
    names = results[0].names  # Tên các lớp dự đoán

    coords = []  # Danh sách lưu trữ tọa độ (x1, y1, x2, y2) và tên đối tượng

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Tọa độ hộp giới hạn
        class_id = int(box.cls[0])  # Lớp ID (danh mục)
        label = names[class_id]  # Lấy tên lớp từ ID
        coords.append({
            'label': label,
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2)
        })

    return annotated_img, coords


def OCR(model, path):
    image, coords = predict_and_extract(model, path)

    # Truy cập từng giá trị trong dictionary coords[0]
    label = coords[0]['label']
    xmin = coords[0]['x1']
    xmax = coords[0]['x2']
    ymin = coords[0]['y1']
    ymax = coords[0]['y2']

    # Crop ROI từ hình ảnh dựa trên tọa độ
    roi = image[int(ymin):int(ymax), int(xmin):int(xmax)]

    # Chuyển đổi ROI sang ảnh grayscale và áp dụng threshold
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_binary = cv2.threshold(roi_gray, 64, 255, cv2.THRESH_BINARY_INV)

    # Đảm bảo rằng roi_binary là một NumPy array
    if not isinstance(roi_binary, np.ndarray):
        raise TypeError('ROI image is not a valid NumPy array')

    # Nhận diện văn bản từ ROI
    text = pt.image_to_string(roi_binary)
    print(text)


# Ví dụ sử dụng hàm:
image_path = 'car.jpg'  # Đường dẫn đến tệp hình ảnh
annotated_img, coords = predict_and_extract(model, image_path)
# OCR(model, image_path)
# Hiển thị ảnh đã được chú thích
cv2.imshow("Predicted Image", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In ra các tọa độ ROI
for coord in coords:
    print(f"Label: {coord['label']}, x1: {coord['x1']}, y1: {coord['y1']}, x2: {coord['x2']}, y2: {coord['y2']}")
