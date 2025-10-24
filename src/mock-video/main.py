import cv2
from pathlib import Path
import time
from injector import Injector

from di import AppModule
from broker import Broker
from stream import VideoStreamer

injector = Injector([AppModule()])

broker = injector.get(Broker)
streamer = injector.get(VideoStreamer)
streamer.start()

video_path = Path("samples/real_05_02_clip_001_1080p_compressed_10fps.mp4")

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise SystemExit(f"Não foi possível abrir: {video_path}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'Frames: {fps}')

if frame_count == 0:
    raise SystemExit("Vídeo sem frames.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        broker.write(frame)
        #cv2.imshow("Video", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break
        time.sleep(0.1)
finally:
    cap.release()
    cv2.destroyAllWindows()
    if hasattr(streamer, "stop"):
        streamer.stop()
