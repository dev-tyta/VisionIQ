import torch
import numpy as np
from src.model.model_loader import ModelLoader
from src.model.config import device, classes, model_confidence
from src.data.image_utils import ImageUtils
from src.data.video_utils import VideoUtils
import cv2


class YoloDetections:
    def __init__(self):
        self.yolo_model = ModelLoader.load_yolo(device)
        self.image_utils = ImageUtils()
        self.video_utils = VideoUtils()
        self.device = device
        self.classes = classes
        self.model_confidence = model_confidence


    def detect_with_yolo(self, image):
        image_handled = self.image_utils.image_handling(image)
        detections = self.yolo_model(image_handled)[0]
        return self.process_yolo_detections(detections)

    def batch_image_detection(self, images):
        images_tensor = torch.stack([self.image_utils.image_handling(image) for image in images]).to(self.device)
        detections = self.yolo_model(images_tensor)[0]
        return self.process_yolo_detections(detections)

    def video_detection(self, video_path):
        video = cv2.VideoCapture(video_path)
        frames = self.video_utils.process_video(video)
        frame_batches = self.video_utils.create_frame_batches(frames)

        all_detections = []
        for batch in frame_batches:
            batch_tensor = torch.stack([self.video_utils.process_video(frame) for frame in batch]).to(self.device)
            detections = self.yolo_model(batch_tensor)[0]
            all_detections.extend(detections)
        
        return all_detections
    
    def process_yolo_detections(self, detections):
        # YOLO detections processing
        processed_detections = []
        for detection in detections:
            scores = detection[5:] # class scores
            class_id = np.argmax(scores)
            class_name = self.classes[class_id]
            confidence = scores[class_id]
            if confidence > self.model_confidence:
                processed_detections.append((class_name, confidence.item()))
        return processed_detections
        

    def people_count(self, detections):
        people = 0
        for i in range(0, len(detections["boxes"])):
            confidence = detections["scores"][i]
            class_idx = int(detections["labels"][i])

            if confidence > self.model_confidence and class_idx == 1:
                label = f"{self.classes[class_idx]}, {class_idx}: {confidence* 100}%"
                print(f"[INFO] {label}")
                people += 1

        return people 

