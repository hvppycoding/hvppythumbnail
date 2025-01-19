import cv2
import numpy as np
from pathlib import Path
import locale
import os
import dlib
import math
from PIL import Image, ImageDraw, ImageFont
import sys

SECONDS_PER_MINUTE = 60

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

def add_filename(frame, filename, padding, timestamp_minutes, timestamp_seconds):
    frame_with_text = np.zeros((frame.shape[0] + padding, frame.shape[1], 3), dtype=np.uint8)
    frame_with_text[:-padding] = frame

    max_width = frame.shape[1] - 10  # 최대 너비 설정

    # 텍스트 생성
    timestamp = f"{timestamp_minutes:02d}:{timestamp_seconds:02d}"  
    text = f"{filename} ({timestamp})"

    pil_img = Image.fromarray(frame_with_text)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("malgun.ttf", 13)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 13)
        except:
            font = ImageFont.load_default()

    # 텍스트 크기 계산 및 파일명 조정
    _x, _y, text_width, text_height = draw.textbbox((0, 0), text, font=font)  # textbbox() 함수 사용

    while text_width > max_width:  # 텍스트 너비가 최대 너비보다 크면
        filename = filename[:-1]  # 파일명 마지막 글자 제거
        text = f"{filename}... ({timestamp})"  # 말줄임표 추가
        _x, _y, text_width, text_height = draw.textbbox((0, 0), text, font=font)  # 텍스트 너비 다시 계산

    # 텍스트 그리기
    draw.text((5, frame.shape[0]), text, font=font, fill=(255, 255, 255))

    return np.array(pil_img)
  
def add_metadata_to_image(image, metadata):
    # PIL 이미지로 변환 후 텍스트 그리기
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    text = metadata
    
    try:
        font = ImageFont.truetype("malgun.ttf", 13)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 13)
        except:
            font = ImageFont.load_default()
            
    max_width = image.shape[1] - 10  # 최대 너비 설정
    _x, _y, text_width, text_height = draw.textbbox((0, 0), text, font=font)  # 텍스트 크기 계산
    
    while text_width > max_width:  # 텍스트 너비가 최대 너비보다 크면
        metadata = metadata[:-1]  # 파일명 마지막 글자 제거
        text = f"{metadata}..."  # 말줄임표 추가
        _x, _y, text_width, text_height = draw.textbbox((0, 0), text, font=font)  # 텍스트 너비 다시 계산

    # 텍스트 위치 계산 (우측 상단)
    x = image.shape[1] - text_width - 10  # 오른쪽 여백 10 픽셀
    y = 10  # 위쪽 여백 10 픽셀
    
    x0 = x
    y0 = y
    x1 = x + text_width
    y1 = y + text_height
    # 배경 영역 어둡게 칠하기
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))  # 투명한 이미지 생성
    draw_overlay = ImageDraw.Draw(overlay)
    draw_overlay.rectangle([(x0, y0), (x1, y1)], fill=(0, 0, 0, 155))  # 반투명 검정색으로 배경 채우기
    pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)  # 배경 이미지와 합성
    
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), text, font=font, fill=(255, 255, 255))  # 흰색으로 텍스트 그리기

    # 다시 OpenCV 이미지로 변환
    image = np.array(pil_img)
    return image

def get_video_duration_seconds(video_path):
    try:
        cap = cv2.VideoCapture(os.path.abspath(str(video_path)))
        if not cap.isOpened():
            return 0
        
        # 프레임 수와 FPS를 이용하여 영상 길이 계산
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps else 0  # fps가 0이면 duration을 0으로 설정
        cap.release()
        return duration
    except:
        return 0
      
def get_video_file_durations_pairs(folder_path):
    folder = Path(folder_path)
    video_files = [f for f in folder.iterdir() if f.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov', '.wmv']]
    
    results = []
    for video_file in video_files:
        results.append((video_file, get_video_duration_seconds(video_file)))
    return results

class VideoCaptureHelper:
  def __init__(self):
    self.current_filename: str = ""
    self.cap: cv2.VideoCapture = None
    
  def is_opened(self):
    return self.cap is not None and self.cap.isOpened()
  
  def release(self):
    if self.is_opened():
      self.cap.release()
      self.cap = None
      self.current_filename = ""
  
  def open(self, filename: str):
    if self.current_filename == filename:
      return True
    if self.is_opened():
      self.release()
    self.cap = cv2.VideoCapture(os.path.abspath(filename))
    if self.cap.isOpened():
      self.current_filename = filename
      return True
    else:
      self.release()
      return False
    
  def get_frame_from_file(self, filename: str, seconds: float):
    if not self.open(filename):
      return None
    total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = self.cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(fps * seconds)
    if frame_idx >= total_frames:
      frame_idx = total_frames - 1
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = self.cap.read()
    if not ret:
      return None
    return frame
  
  def __del__(self):
    self.release()
  
class VideoDurationDivider:
  # 여러 영상을 하나의 영상 파일로 합쳤다고 가정했을 때
  # 특정 초에 해당하는 영상 파일과 그 영상 파일에서의 시간을 계산하는 클래스
  def __init__(self, video_file_duration_pairs):
    self.video_file_duration_pairs = video_file_duration_pairs
    self.sorter = lambda x: x[0]
    self.video_file_duration_pairs.sort(key=self.sorter)
    self.total_duration = sum([duration for _, duration in video_file_duration_pairs])
    
  def get_video_file_and_section(self, seconds):
    if seconds < 0 or seconds >= self.total_duration:
      return None, None
    current_duration = 0
    for video_file, duration in self.video_file_duration_pairs:
      if current_duration + duration > seconds:
        return video_file, seconds - current_duration
      current_duration += duration
    return None, None
  
def calculate_midpoints(start, end, num_sections):
  """
  주어진 범위를 지정된 개수의 영역으로 나누고 각 영역의 중간값을 계산합니다.

  Args:
    start: 범위의 시작 값.
    end: 범위의 끝 값.
    num_sections: 나눌 영역의 개수.

  Returns:
    각 영역의 중간값을 담은 리스트.
  """
  step = (end - start) / num_sections
  midpoints = []
  for i in range(num_sections):
    section_start = start + i * step
    section_end = section_start + step
    midpoint = (section_start + section_end) / 2
    midpoints.append(midpoint)
  return midpoints

def resize_image(frame, target_width, target_height):
    height, width = frame.shape[:2]
    height_ratio = target_height / height
    width_ratio = target_width / width
    if height_ratio < width_ratio:
        new_height = target_height
        new_width = int(width * height_ratio)
    else:
        new_width = target_width
        new_height = int(height * width_ratio)
    return cv2.resize(frame, (new_width, new_height))
  
class FolderThumbnailCreator:
  def __init__(self, folder_path, grid_height=3, grid_width=4, capture_height=480, capture_width=480, filename_height=30, nstep_per_capture=10):
    self.folder_path = folder_path
    self.grid_height = grid_height
    self.grid_width = grid_width
    self.capture_height = capture_height
    self.capture_width = capture_width
    self.filename_height = filename_height
    self.nstep_per_capture = nstep_per_capture
    self.video_file_duration_pairs = get_video_file_durations_pairs(folder_path)
    self.total_duration = sum([duration for _, duration in self.video_file_duration_pairs])
    self.total_sections = grid_height * grid_width
    self.duration_per_section = self.total_duration / self.total_sections
    self.duration_divider = VideoDurationDivider(self.video_file_duration_pairs)
    self.video_capture_helper = VideoCaptureHelper()
    
  def find_thumbnail_for_section(self, section):
    section_start_time = section * self.duration_per_section
    section_end_time = (section + 1) * self.duration_per_section
    midpoints = calculate_midpoints(section_start_time, section_end_time, self.nstep_per_capture)
    for midpoint in midpoints:
        video_file, section_time = self.duration_divider.get_video_file_and_section(midpoint)
        if video_file is None:
            print("Video file not found!")
            continue
        frame = self.video_capture_helper.get_frame_from_file(str(video_file), section_time)
        if frame is None:
            print("Frame not found!")
            continue
        frame = resize_image(frame, self.capture_width, self.capture_height)
        has_faces, _, frame_with_faces = detect_faces(frame)
        if has_faces:
            return frame_with_faces, video_file, section_time
    middle_time = (section_start_time + section_end_time) / 2
    video_file, section_time = self.duration_divider.get_video_file_and_section(middle_time)
    frame = self.video_capture_helper.get_frame_from_file(str(video_file), section_time)
    frame = resize_image(frame, self.capture_width, self.capture_height)
    return frame, video_file, section_time
  
  def create_thumbnail_for_section(self, section):
    frame, video_file, section_time = self.find_thumbnail_for_section(section)

    if frame is not None:
        canvas = np.zeros((self.capture_height, self.capture_width, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        y_offset = (self.capture_height - h) // 2
        x_offset = (self.capture_width - w) // 2
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = frame
        minutes = int(section_time) // SECONDS_PER_MINUTE
        seconds = int(section_time) % SECONDS_PER_MINUTE
        frame_with_text = add_filename(canvas, video_file.name, self.filename_height, minutes, seconds)
    return frame_with_text
  
  def generate_metadata_string(self):
    total_duration_minutes = int(self.total_duration) // SECONDS_PER_MINUTE
    total_num_videos = len(self.video_file_duration_pairs)
    total_size = sum([video_file.stat().st_size for video_file, _ in self.video_file_duration_pairs])
    total_size_in_mb = total_size / 1024 / 1024
    if total_size_in_mb >= 1024:
        total_size_in_gb = total_size_in_mb / 1024
        return f"#Videos: {total_num_videos}, Total Duration: {total_duration_minutes} min., Total Size: {total_size_in_gb:.1f} GB"
    return f"#Videos: {total_num_videos}, Total Duration: {total_duration_minutes} min., Total Size: {total_size_in_mb:.1f} MB"
    
  def create_thumbnail(self, output_path):
    frames = []
    for section in range(self.total_sections):
        frame = self.create_thumbnail_for_section(section)
        frames.append(frame)
    rows = []
    for i in range(0, len(frames), self.grid_width):
        row = np.hstack(frames[i:i + self.grid_width])
        rows.append(row)
    
    final_image = np.vstack(rows)
    
    final_image = add_metadata_to_image(final_image, self.generate_metadata_string())
    output_path = os.path.abspath(output_path)
    try:
        import imageio
        imageio.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        return True
    except Exception as e:
        print(f"저장 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    folder_path = sys.argv[1]
    output_path = "thumbnail.jpg"
    FolderThumbnailCreator(folder_path).create_thumbnail(output_path)