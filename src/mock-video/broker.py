import queue
import cv2
import base64

class Broker(queue.Queue):
    def __init__(self):
        super().__init__()
        
    def write(self, frame):
        if frame is not None and frame.size > 0:
            if not self.empty():
                while not self.empty():
                    self.get_nowait()
            
            success, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            self.put_nowait(f'data: {base64.b64encode(encoded_frame).decode('utf-8')}\n\n')

    def __iter__(self):
        return iter(self.get, None)