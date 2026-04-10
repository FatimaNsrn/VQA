from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "blip_vqa_finetuned"

processor = BlipProcessor.from_pretrained(MODEL_PATH)
model = BlipForQuestionAnswering.from_pretrained(MODEL_PATH).to(device)

def predict(image_path, question):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, question, return_tensors="pt").to(device)

    output = model.generate(**inputs)
    answer = processor.decode(output[0], skip_special_tokens=True)

    return answer