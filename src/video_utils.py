import cv2
from torchvision.transforms import functional as F

class VideoUtils:
    def __init__(self):
        pass

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = F.to_tensor(frame)
        return frame
    
    def process_video(self, video_path):
        video = cv2.VideoCapture(video_path)
        frames = []

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = self.process_frame(frame)
            frames.append(frame)
        
        video.release()
        return frames
    