import os
from ultralytics import YOLO

model = YOLO("src/runs/detect/train4/weights/best.pt")

image_folder = "src/placas_dataset/tests/images"

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if image_path.endswith((".jpg", ".png", ".jpeg")):
        results = model(image_path, conf=0.5)
        results[0].show()