import base64, io
from PIL import Image
import os

def generate_and_save_image(imagebase64: str) -> str:
    
    split = imagebase64.split(",")
    img_extension = split[0].split(";")[0].split("/")[1]
    image = split[1]

    filename = f"new_image.{img_extension}"
    image_path = os.path.join(os.path.dirname(__file__), filename)
    img_to_save = Image.open(io.BytesIO(base64.decodebytes(bytes(image, "utf-8"))))
    img_to_save.save(image_path, quality=100, subsampling=0)

    return image_path