#!/usr/bin/env python3
"""
Script to rename all files in v4/dataset subdirectories to shorter numbered names.
Renames files to 0.png, 1.png, 2.png, etc. in each subdirectory.
"""

import os
import pathlib
from pathlib import Path


def rename_files_in_directory(directory_path):
    """
    Rename all PNG files in a directory to numbered files (0.png, 1.png, etc.)
    
    Args:
        directory_path (Path): Path to the directory containing files to rename
    """
    # Get all PNG files in the directory
    png_files = list(directory_path.glob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in {directory_path}")
        return
    
    # Sort files to ensure consistent ordering
    png_files.sort()
    
    print(f"Found {len(png_files)} PNG files in {directory_path}")
    
    # Create a temporary mapping to avoid conflicts during renaming
    temp_renames = []
    
    # First pass: rename to temporary names
    for i, file_path in enumerate(png_files):
        temp_name = directory_path / f"temp_{i}.png"
        temp_renames.append((file_path, temp_name, i))
        
    # Rename to temporary names first
    for original, temp, index in temp_renames:
        original.rename(temp)
        
    # Second pass: rename from temporary to final names
    for original, temp, index in temp_renames:
        final_name = directory_path / f"{index}.png"
        temp.rename(final_name)
        print(f"  Renamed {original.name} -> {index}.png")

def main():
    """
    Main function to process all subdirectories in v4/dataset
    """
    dataset_path = Path("v4/dataset")
    
    if not dataset_path.exists():
        print(f"Error: {dataset_path} does not exist!")
        return
    
    print(f"Processing dataset directory: {dataset_path}")
    
    # Process train and test directories
    for split_dir in ["train", "test"]:
        split_path = dataset_path / split_dir
        
        if not split_path.exists():
            print(f"Skipping {split_path} - directory does not exist")
            continue
            
        print(f"\nProcessing {split_path}...")
        
        # Get all subdirectories (font directories)
        font_directories = [d for d in split_path.iterdir() if d.is_dir()]
        
        print(f"Found {len(font_directories)} font directories in {split_path}")
        
        for font_dir in font_directories:
            print(f"\nRenaming files in {font_dir.name}...")
            rename_files_in_directory(font_dir)
    
    print("\nFile renaming complete!")

if __name__ == "__main__":
    main() 