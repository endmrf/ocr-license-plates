from typing import NamedTuple
from ultralytics import YOLO
import cv2
import easyocr
import os

class OcrEasyOcrParameter(NamedTuple):
    image_path: str

class OcrEasyOcrUseCase():

    def __init__(self, model: YOLO):
        self.model = model
        self.reader = None
    
    def _get_reader(self):
        if self.reader is None:
            self.reader = easyocr.Reader(['pt', 'en'])
        return self.reader

    def execute(self, parameter: OcrEasyOcrParameter) -> str:

        if not parameter.image_path:
            return {"success": False, "text": "Caminho da imagem não fornecido"}
        
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

                # # Salvar a imagem cortada
                # original_filename = os.path.basename(parameter.image_path)
                # filename_without_ext = os.path.splitext(original_filename)[0]
                # cropped_filename = f"{filename_without_ext}_cropped.jpg"
                # cv2.imwrite(cropped_filename, cropped)
                # # print(f"Imagem cortada salva em: {cropped_filename}")

                result = self._get_reader().readtext(cropped, detail=0, paragraph=False)
                text = result[0].strip() if result else "NÃO DETECTADO"

                return {"success": True, "text": text}
            
            return {"success": False, "text": "Nenhuma placa detectada"}
        
        except Exception as e:
            print(f"Erro ao processar a imagem: {str(e)}")
            return {"success": False, "text": None}