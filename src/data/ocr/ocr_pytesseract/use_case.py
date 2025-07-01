from typing import NamedTuple
from ultralytics import YOLO
import cv2
import pytesseract

class OcrPytesseractParameter(NamedTuple):
    image_path: str

class OcrPytesseractUseCase:

    def __init__(self, model: YOLO):
        self.model = model

    def execute(self, parameter: OcrPytesseractParameter) -> str:
        
        if not parameter.image_path:
            return {
                "success": False, 
                "text": None,
                "message": "Caminho da imagem não fornecido"
            }

        try:

            image = cv2.imread(parameter.image_path)

            results = self.model(parameter.image_path, conf=0.5)

            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Calcular nova área do recorte eliminando a parte superior (onde fica "BRASIL")
                height = y2 - y1
                # Remove aproximadamente 35% da parte superior da placa
                new_y1 = y1 + int(height * 0.30)
                
                cropped = image[new_y1:y2, x1:x2]

                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                text = pytesseract.image_to_string(thresh, config='--psm 8')

                if text:
                    return {"success": True, "text": text.strip().replace("\n", "")}
                
                return "NÃO DETECTADO"

            return {"success": False, "text": "Nenhuma placa detectada"}
        
        except Exception as e:
            print(f"Erro ao processar a imagem: {str(e)}")
            return {
                "success": False, 
                "text": None,
            }