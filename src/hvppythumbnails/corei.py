import cv2
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import locale
import os
import dlib
import math
from PIL import Image, ImageDraw, ImageFont
import sys
import random

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

def create_and_merge_thumbnails(image_dir, output_filename, grid_width=4, grid_height=3, thumbnail_width=480, thumbnail_height=480):
  """
  OpenCV를 사용하여 이미지 조건을 판별하고, 조건에 따라 우선순위를 정하여 
  썸네일을 생성하고 합칩니다.

  Args:
    image_dir: 이미지가 있는 폴더 경로
    grid_size: (rows, cols) 형태의 grid 크기
    output_filename: 합쳐진 이미지를 저장할 파일 이름
  """
  import os

  rows, cols = (grid_height, grid_width)  # grid 크기 (예: 3x4)
  thumbnail_size = (thumbnail_width, thumbnail_height)  # 썸네일 크기 (예: 100x100)
  new_image = Image.new('RGB', (cols * thumbnail_size[0], rows * thumbnail_size[1]))

  img_list = []
  n_face_imgs = 0
  n_final_imgs = grid_height * grid_width
  random_list_dir = list(os.listdir(image_dir))
  random.shuffle(random_list_dir)
  for filename in random_list_dir:
    try:
      # 이미지 파일 읽기
      img = Image.open(os.path.join(image_dir, filename))
      img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
      has_faces, _, img = detect_faces(img)
      if has_faces:
        img_list.insert(0, img)  # 우선순위 높음
        n_face_imgs += 1
        if n_face_imgs >= n_final_imgs:
          break
      else:
        img_list.append(img)

    except cv2.error:
      print(f"{filename}은 이미지 파일이 아닙니다.")

  img_list = img_list[:rows * cols]  # grid 크기만큼 이미지 추출
  
  if len(img_list) < rows * cols:
    img_list = (img_list * (rows * cols // len(img_list) + 1))[:rows * cols]


  for i, img in enumerate(img_list):
    row = i // cols
    col = i % cols

    # 이미지 비율 유지하며 리사이즈
    height, width, _ = img.shape
    ratio_width = thumbnail_width / width
    ratio_height = thumbnail_height / height
    if ratio_height < ratio_width:
        new_height = thumbnail_height
        new_width = int(width * (new_height / height))
    else:
        new_width = thumbnail_width
        new_height = int(height * (new_width / width))
    img = cv2.resize(img, (new_width, new_height))

    # cv2 이미지를 PIL 이미지로 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # 썸네일 이미지를 캔버스 중앙에 배치
    x_offset = int((thumbnail_width - new_width) / 2)
    y_offset = int((thumbnail_height - new_height) / 2)
    new_image.paste(img, (col * thumbnail_width + x_offset, row * thumbnail_height + y_offset))

  new_image.save(output_filename)
