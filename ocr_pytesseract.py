import os
import cv2
import pytesseract
from ultralytics import YOLO

model = YOLO("src/runs/detect/train4/weights/best.pt")

INPUT_FOLDER = "src/placas_dataset/tests/images"
OUTPUT_FOLDER = "src/placas_dataset/results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(INPUT_FOLDER, filename)
    image = cv2.imread(image_path)

    results = model(image_path, conf=0.5)

    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box[:4])
        cropped = image[y1:y2, x1:x2]

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        texto = pytesseract.image_to_string(thresh, config='--psm 8')
        texto_limpo = texto.strip().replace("\n", "")

        placa_img_name = f"{os.path.splitext(filename)[0]}_placa_{i+1}.jpg"
        placa_txt_name = f"{os.path.splitext(filename)[0]}_placa_{i+1}.txt"

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, placa_img_name), cropped)

        with open(os.path.join(OUTPUT_FOLDER, placa_txt_name), "w") as f:
            f.write(texto_limpo)

        print(f"[{filename}] Placa detectada ({i+1}): {texto_limpo}")