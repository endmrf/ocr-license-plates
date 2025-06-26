from typing import NamedTuple
from ultralytics import YOLO
import cv2
import easyocr

class OcrEasyOcrParameter(NamedTuple):
    image_path: str

class OcrEasyOcrUseCase():

    def __init__(self, model: YOLO):
        self.model = model
        self.reader = easyocr.Reader(['pt', 'en'])

    def execute(self, parameter: OcrEasyOcrParameter) -> str:

        if not parameter.image_path:
            return {"success": False, "text": "Caminho da imagem não fornecido"}
        
        try:
            image = cv2.imread(parameter.image_path)

            results = self.model(parameter.image_path, conf=0.5)
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped = image[y1:y2, x1:x2]

                result = self.reader.readtext(cropped, detail=0, paragraph=False)
                text = result[0].strip() if result else "NÃO DETECTADO"

                return {"success": True, "text": text}
            
            return {"success": False, "text": "Nenhuma placa detectada"}
        
        except Exception as e:
            print(f"Erro ao processar a imagem: {str(e)}")
            return {"success": False, "text": None}