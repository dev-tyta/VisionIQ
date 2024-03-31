# making a test for the project
from fasterrcnn_detections import Detections
from yolo_detections import YoloDetections

image_path = 'data/p.jpg'
detection = Detections()
yolo = YoloDetections()

output = yolo.detect_with_yolo(image_path)
counting = yolo.people_count(output)

print(f"[INFO] People in the image: {counting}")