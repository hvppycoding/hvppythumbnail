import cv2
import numpy as np
from pathlib import Path
import locale
import os
import dlib
import math
from PIL import Image, ImageDraw, ImageFont

def detect_faces(frame):
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    frame_with_faces = frame.copy()
    for face in faces:
        cv2.rectangle(frame_with_faces, 
                     (face.left(), face.top()),
                     (face.right(), face.bottom()),
                     (0, 255, 0), 2)
    return len(faces) > 0, len(faces), frame_with_faces

def add_filename(frame, filename, width):
    padding = 30
    frame_with_text = np.zeros((frame.shape[0] + padding, frame.shape[1], 3), dtype=np.uint8)
    frame_with_text[:-padding] = frame
    
    if len(filename) > 30:
        filename = filename[:27] + "..."
        
    pil_img = Image.fromarray(frame_with_text)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("malgun.ttf", 13)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 13)
        except:
            font = ImageFont.load_default()
            
    draw.text((5, frame.shape[0]), filename, font=font, fill=(255, 255, 255))
    return np.array(pil_img)

def extract_video_frame(video_path, section_start, section_end):
    try:
        cap = cv2.VideoCapture(os.path.abspath(str(video_path)))
        if not cap.isOpened():
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        nstep = 20
        step = (section_end - section_start) / nstep
        
        for pos in np.arange(section_start, section_end, step):
            frame_idx = int(total_frames * pos)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                height, width = frame.shape[:2]
                if height > width:
                    new_height = 480
                    new_width = int(width * (480 / height))
                else:
                    new_width = 480
                    new_height = int(height * (480 / width))
                
                frame = cv2.resize(frame, (new_width, new_height))
                has_faces, _, frame_with_faces = detect_faces(frame)
                if has_faces:
                    cap.release()
                    return frame_with_faces
        
        middle_pos = (section_start + section_end) / 2
        frame_idx = int(total_frames * middle_pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            height, width = frame.shape[:2]
            if height > width:
                new_height = 480
                new_width = int(width * (480 / height))
            else:
                new_width = 480
                new_height = int(height * (480 / width))
            return cv2.resize(frame, (new_width, new_height))
        return None
    except Exception as e:
        print(f"Frame extraction error: {e}")
        return None

def get_video_files(folder_path, max_videos=9):
    folder = Path(folder_path)
    video_files = [f for f in folder.iterdir() if f.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov', '.wmv']]
    
    if len(video_files) > max_videos:
        video_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        return video_files[:max_videos]
    elif len(video_files) == 1:
        return [video_files[0]] * max_videos
    else:
        return video_files * (max_videos // len(video_files) + 1)

def calculate_sections_per_video(total_slots, video_count, video_index):
    if video_count == 1:
        return total_slots
    
    remaining_slots = total_slots
    remaining_videos = video_count
    
    for i in range(video_index):
        slots = math.ceil(remaining_slots / remaining_videos)
        remaining_slots -= slots
        remaining_videos -= 1
    
    return math.ceil(remaining_slots / remaining_videos)

def get_section_bounds(video_index, file_count, current_video_sections):
    section_size = 1.0 / current_video_sections
    return section_size * (video_index % current_video_sections), section_size * ((video_index % current_video_sections) + 1)

def create_folder_thumbnail(folder_path, output_path, max_videos=9):
    video_files = get_video_files(folder_path, max_videos)
    file_count = len(set(str(f) for f in video_files))
    
    frames = []
    current_index = 0
    
    for video_index, video in enumerate(set(video_files)):
        sections = calculate_sections_per_video(max_videos, file_count, video_index)
        for section in range(sections):
            section_start, section_end = get_section_bounds(section, file_count, sections)
            frame = extract_video_frame(video, section_start, section_end)
            if frame is not None:
                canvas = np.zeros((480, 480, 3), dtype=np.uint8)
                h, w = frame.shape[:2]
                y_offset = (480 - h) // 2
                x_offset = (480 - w) // 2
                canvas[y_offset:y_offset+h, x_offset:x_offset+w] = frame
                canvas = add_filename(canvas, video.name, w)
                frames.append(canvas)
                current_index += 1
    
    if not frames:
        return False
        
    grid_size = 3
    while len(frames) < grid_size * grid_size:
        frames.append(np.zeros((480, 480, 3), dtype=np.uint8))
    
    rows = []
    for i in range(0, len(frames), grid_size):
        row = np.hstack(frames[i:i + grid_size])
        rows.append(row)
    
    final_image = np.vstack(rows)
    
    output_path = os.path.abspath(output_path)
    try:
        import imageio
        imageio.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        return True
    except Exception as e:
        print(f"저장 중 오류 발생: {e}")
        return False