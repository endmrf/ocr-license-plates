import os
from flask import Blueprint, request, jsonify, g
from flask_cors import CORS
from ultralytics import YOLO
from .utils import generate_and_save_image
from src.data.ocr.ocr_easyocr import (
    OcrEasyOcrUseCase,
    OcrEasyOcrParameter
)
from src.data.ocr.ocr_pytesseract import (
    OcrPytesseractUseCase,
    OcrPytesseractParameter
)

bp = Blueprint('main', __name__)
CORS(bp)

model = YOLO("src/ml/runs/detect/train4/weights/best.pt")

@bp.route('/', methods=['GET'])
def index():
    return (
    """
        Bem-vindo(a)!,
        Utilize /ocr/easyocr ou /ocr/pytesseract.
    """
    )

@bp.route('/ocr/easyocr', methods=['POST'])
def process_image_easyocr():
    use_case = OcrEasyOcrUseCase(model)
    parameter = OcrEasyOcrParameter(
        image_path=g.filename,
    )
    response = use_case.execute(parameter)
    return jsonify(response)

@bp.route('/ocr/pytesseract', methods=['POST'])
def process_image_pytesseract():
    use_case = OcrPytesseractUseCase(model)
    parameter = OcrPytesseractParameter(
        image_path=g.filename,
    )
    response = use_case.execute(parameter)
    return jsonify(response)

@bp.before_request
def before_request():
    data = request.get_json()
    image = data.get('image', '')
    if not image:
        return jsonify({"error": "Image data is required"}), 400
    
    filename = generate_and_save_image(
        image
    )
    
    g.filename = filename


@bp.after_request
def after_request(response):
    if hasattr(g, 'filename'):
        try:
            os.remove(g.filename)
            
        except Exception as e:
            print(f"Error removing file {g.filename}: {str(e)}")

    return response