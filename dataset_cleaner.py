import os
from PIL import Image
import sys
from pathlib import Path
from tqdm import tqdm

def is_valid_image(file_path: Path) -> bool:
    try:
        with Image.open(file_path) as img:
            # Try to load the image to verify it's not corrupted
            img.verify()
            # Try to load it again to check if it can be processed
            img = Image.open(file_path)
            img.load()
        return True
    except Exception as e:
        return False

def count_image_files(directory: Path) -> int:
    count: int = 0
    for _, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}:
                count += 1
    return count

def check_images_in_directory(directory: Path) -> None:
    # Count total image files first
    total_files: int = count_image_files(directory)
    
    # Get all files recursively with progress bar
    with tqdm(total=total_files, desc="Checking images") as pbar:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                # Check if file is an image (basic extension check)
                if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}:
                    if not is_valid_image(file_path):
                        print(f"Malformed image found: {file_path}")
                    pbar.update(1)

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python dataset_cleaner.py <directory_path>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    print(f"Checking images in {directory}...")
    check_images_in_directory(directory)
    print("Image check complete.")

if __name__ == "__main__":
    main()
