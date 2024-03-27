# making a test for the project
from detections import Detections

image_detection = Detections.image_detection()
image_path = '../data/fam1.HEIC'

output = image_detection(image_path)

print(output)