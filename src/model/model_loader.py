import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as fastercnn_model
from transformers import pipeline
from config import device

class ModelLoader:
    def __init__(self):
        self.device = device

    # def load_fastercnn(self):
    #     try:
    #         model = fastercnn_model(pretrained=True, progress=True, pretrained_backbone=True)
    #         model.to(self.device)
    #         model.eval()
    #         return model
    #     except Exception as e:
    #         print(f"Failed to load Faster R-CNN model: {e}")
    #         return None

    # def load_yolo(self, model_name="yolov5s"):
    #     try:
    #         model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
    #         model.to(self.device)
    #         model.eval()
    #         return model
    #     except Exception as e:
    #         print(f"Failed to load YOLO model: {e}")
    #         return None

    def load_llava(self):
        try:
            # llava_processor = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=None)
            # llava_model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=None)
            llava_model = pipeline("feature-extraction", model="llava-hf/llava-v1.6-mistral-7b-hf", device=self.device)
            llava_model.to(self.device)
            llava_model.eval()
            return llava_processor, llava_model
        except Exception as e:
            print(f"Failed to load LLaVA model: {e}")
            return None, None

model = ModelLoader()
# faster_rcnn_model = model.load_fastercnn()
# yolo_model = model.load_yolo()
llava_processor, llava_model = model.load_llava()

# Use some print statements or logging to check the models loaded
# print("Faster R-CNN Model Loaded:", faster_rcnn_model is not None)
# print("YOLO Model Loaded:", yolo_model is not None)
print("LLaVA Model and Processor Loaded:", llava_model is not None)
