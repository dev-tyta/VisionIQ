# making a test for the project

from src.count.yolo_detections import YoloDetections


image_path = '/workspaces/VisionIQ/data/acc.jpg'
yolo_model = YoloDetections()


counting = yolo_model.detect_with_yolo(image=image_path)

print(f"[INFO] People in the image: {counting}")
