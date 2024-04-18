# making a test for the project
from src.model.yolo_detections import YoloDetections

image_path = 'data/p.jpg'
yolo = YoloDetections()

output = yolo.detect_with_yolo(image_path)
counting = yolo.people_count(output)

print(f"[INFO] People in the image: {counting}")