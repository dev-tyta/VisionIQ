import cv2
from torchvision.transforms import functional as F

class VideoUtils:
    def __init__(self):
        pass

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = F.to_tensor(frame)
        return frame
    
    def process_video(self, video):
        frames = []

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = self.process_frame(frame)
            frames.append(frame)
        
        video.release()
        return frames
    
    def create_frame_batches(self, frames, batch_size=16):
        """
        Splits the list of frames into smaller lists of frames (batches), each with a size up to `batch_size`.
        
        :param frames: A list of video frames.
        :param batch_size: The maximum number of frames per batch.
        :return: A list of batches, where each batch is a list of frames.
        """
        # Split the frames list into batches of size `batch_size`
        for i in range(0, len(frames), batch_size):
            yield frames[i:i + batch_size]
    