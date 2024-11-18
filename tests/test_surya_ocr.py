from PIL import Image

from surya.model.detection.model import load_model as load_det_model
from surya.model.detection.model import load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.ocr import run_ocr

IMAGE_PATH = "./img/Snipaste_2024-10-20_23-09-41.png"
image = Image.open(IMAGE_PATH)
langs = ["en"]  # Replace with your languages - optional but recommended
det_processor, det_model = load_det_processor(), load_det_model()
rec_model, rec_processor = load_rec_model(), load_rec_processor()

predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)
print(predictions)
