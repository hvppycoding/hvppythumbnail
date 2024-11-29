import cv2
import numpy as np
from pathlib import Path

def extract_video_frame(video_path, frame_position=0.2):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = int(total_frames * frame_position)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.resize(frame, (320, 180))
    return None

def create_folder_thumbnail(folder_path, output_path, max_videos=9):
    folder = Path(folder_path)
    video_files = [f for f in folder.iterdir() if f.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov', '.wmv']]
    video_files = video_files[:max_videos]
    
    frames = []
    for video in video_files:
        frame = extract_video_frame(video)
        if frame is not None:
            frames.append(frame)
    
    if not frames:
        return False
        
    grid_size = int(np.ceil(np.sqrt(len(frames))))
    while len(frames) < grid_size * grid_size:
        frames.append(np.zeros_like(frames[0]))
    
    rows = []
    for i in range(0, len(frames), grid_size):
        row = np.hstack(frames[i:i + grid_size])
        rows.append(row)
    
    final_image = np.vstack(rows)
    cv2.imwrite(str(output_path), final_image)
    return True