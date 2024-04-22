# importing libraries
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as fastercnn_model
from transformers import LlavaNextForConditionalGeneration
from src.model.config import device


class ModelLoader:
    def __init__(self):
        self.device = device
        
    def load_fastercnn(device):
        # loading Faster RCNN ResNet50 model
        model = fastercnn_model(pretrained=True, progress=True, pretrained_backbone=True)
        model.to(device)  
        model.eval() # prints out the architecture of the model
        return model
    
    def load_yolo(device, model_name= "yolov5s"):
        model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True, trust_repo=True)
        model.to(device)
        model.eval()
        return model 
    
    def load_llava(self):
        llava_processor = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                                        torch_dtype=torch.float16,
                                                                        low_cpu_mem_usage=True,
                                                                        use_flash_attention_2=True
                                                                        )
        llava_model.to(self.device)
        llava_model.eval()
        return llava_processor, llava_model
    
