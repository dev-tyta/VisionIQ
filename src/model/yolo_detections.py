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
        detections = self.yolo_model(image)
        return self.process_yolo_detections(detections)

    def batch_image_detection(self, images):
        images_tensor = torch.stack(images).to(self.device)
        detections = self.yolo_model(images_tensor)

        all_processed_detections = []
        for detection in detections.xyxy:
            processed = self.process_yolo_detections(detection)
            all_processed_detections.append(processed)

        return all_processed_detections

    def video_detection(self, video_path):
        video = cv2.VideoCapture(video_path)
        frames = self.video_utils.process_video(video)
        frame_batches = self.video_utils.create_frame_batches(frames)

        all_detections = []
        for batch in frame_batches:
            batch_tensor = torch.stack(batch).to(self.device)
            detections = self.yolo_model(batch_tensor)[0]
            
            for det in detections.xyxy:
                all_detections.extend(det.cpu().numpy())
        
        return all_detections
    
    def process_yolo_detections(self, detections):
        # YOLO detections processing
        processed_detections = []
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            class_id = int(class_id)
            class_name = self.classes[class_id]
            if confidence > self.model_confidence:
                processed_detections.append((class_name, confidence.item()))
        return processed_detections
        

    def people_count(self, detections):
        people = 0
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            class_id = int(class_id)

            if confidence > self.model_confidence and class_id == 0:
                label = f"{self.classes[class_id]}, {class_id}: {confidence* 100:.2f}%"
                print(f"[INFO] {label}")
                people += 1

        return people 

