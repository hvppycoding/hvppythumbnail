import argparse
from pathlib import Path
from .core import create_folder_thumbnail

def main():
    parser = argparse.ArgumentParser(description='비디오 폴더 썸네일 생성기')
    parser.add_argument('folder_path', help='비디오가 있는 폴더 경로')
    parser.add_argument('-o', '--output', help='출력 이미지 경로 (기본: 현재 디렉토리/thumbnail.jpg)')
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        print(f"오류: 폴더를 찾을 수 없습니다 - {folder_path}")
        return
    
    output_path = args.output if args.output else Path.cwd() / "thumbnail.jpg"
    
    if create_folder_thumbnail(folder_path, output_path):
        print(f"썸네일 생성 완료: {output_path}")
    else:
        print("오류: 썸네일을 생성할 수 없습니다.")

if __name__ == "__main__":
    main()