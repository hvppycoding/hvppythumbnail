import argparse
from pathlib import Path
from .core import FolderThumbnailCreator
from .corei import create_and_merge_thumbnails

def main():
    parser = argparse.ArgumentParser(description='비디오 폴더 썸네일 생성기')
    parser.add_argument('folder_path', help='비디오가 있는 폴더 경로')
    parser.add_argument('-o', '--output', help='출력 이미지 경로 (기본: 현재 디렉토리/thumbnail.jpg)')
    parser.add_argument("--grid_height", type=int, default=3, help="썸네일 그리드 높이")
    parser.add_argument("--grid_width", type=int, default=4, help="썸네일 그리드 너비")
    parser.add_argument("--capture_height", type=int, default=480, help="캡처 이미지 높이")
    parser.add_argument("--capture_width", type=int, default=480, help="캡처 이미지 너비")
    parser.add_argument("--filename_height", type=int, default=30, help="파일 이름 높이")
    parser.add_argument("--nstep_per_capture", type=int, default=25, help="캡처 빈도")
    parser.add_argument("--image", action="store_true", help="이미지 파일을 사용")
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        print(f"오류: 폴더를 찾을 수 없습니다 - {folder_path}")
        return
    
    output_path = args.output if args.output else Path.cwd() / "thumbnail.jpg"
    
    if args.image:
        create_and_merge_thumbnails(folder_path, output_path, 
                                    grid_width=args.grid_width, 
                                    grid_height=args.grid_height, 
                                    thumbnail_width=args.capture_width, 
                                    thumbnail_height=args.capture_height)
        return
    try:
        c = FolderThumbnailCreator(folder_path, 
                                   grid_height=args.grid_height, 
                                   grid_width=args.grid_width, 
                                   capture_height=args.capture_height, 
                                   capture_width=args.capture_width,
                                   filename_height=args.filename_height, 
                                   nstep_per_capture=args.nstep_per_capture)
        c.create_thumbnail(output_path)
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    main()