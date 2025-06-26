import os
import cv2
from ultralytics import YOLO
import easyocr

model = YOLO("src/runs/detect/train4/weights/best.pt")
reader = easyocr.Reader(['pt', 'en'])

INPUT_FOLDER = "src/placas_dataset/tests/images"
OUTPUT_FOLDER = "src/placas_dataset/results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(INPUT_FOLDER, filename)
    image = cv2.imread(image_path)

    results = model(image_path, conf=0.5)

    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box[:4])
        print(image[y1:y2, x1:x2])
        cropped = image[y1:y2, x1:x2]

        result = reader.readtext(cropped, detail=0, paragraph=False)
        texto = result[0].strip() if result else "NÃO DETECTADO"

        placa_img = f"{os.path.splitext(filename)[0]}_placa_{i+1}.jpg"
        placa_txt = f"{os.path.splitext(filename)[0]}_placa_{i+1}.txt"

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, placa_img), cropped)
        with open(os.path.join(OUTPUT_FOLDER, placa_txt), "w") as f:
            f.write(texto)

        print(f"[{filename}] Placa detectada ({i+1}): {texto}")