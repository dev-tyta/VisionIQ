# making a test for the project
from fasterrcnn_detections import Detections

image_path = 'data/p.jpg'
detection = Detections()

output = detection.image_detection(image_path)
counting = detection.people_count(output)

print(f"[INFO] People in the image: {counting}")