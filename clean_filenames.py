#!/usr/bin/env python3
"""
Script to rename files in a directory by sanitizing their filename stems.
Uses the sanitize_filename function to replace special characters with safe alternatives.
"""

import argparse
import logging
import os
import pathlib
import sys
from typing import List, Tuple

# Import the sanitize_filename function from dataset_generator
from dataset_generator import sanitize_filename

logger = logging.getLogger(__name__)

def get_files_to_rename(directory: pathlib.Path, recursive: bool = False) -> List[Tuple[pathlib.Path, str, str]]:
    """
    Get list of files that need to be renamed.
    
    Returns:
        List of tuples: (file_path, original_stem, sanitized_stem)
    """
    files_to_rename = []
    
    if recursive:
        # Get all files recursively
        pattern = "**/*"
        files = directory.glob(pattern)
    else:
        # Get files only in the current directory
        files = directory.iterdir()
    
    for file_path in files:
        if file_path.is_file():
            original_stem = file_path.stem
            sanitized_stem = sanitize_filename(original_stem)
            
            # Only include files that need renaming
            if original_stem != sanitized_stem:
                files_to_rename.append((file_path, original_stem, sanitized_stem))
    
    return files_to_rename

def rename_files(directory: pathlib.Path, recursive: bool = False, dry_run: bool = False) -> None:
    """
    Rename files in the directory by sanitizing their filename stems.
    
    Args:
        directory: Directory containing files to rename
        recursive: Whether to process subdirectories recursively
        dry_run: If True, only show what would be renamed without actually renaming
    """
    if not directory.exists():
        raise ValueError(f"Directory {directory} does not exist")
    
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")
    
    # Get files that need renaming
    files_to_rename = get_files_to_rename(directory, recursive)
    
    if not files_to_rename:
        logger.info("No files need renaming")
        return
    
    logger.info(f"Found {len(files_to_rename)} files that need renaming")
    
    # Track statistics
    renamed_count = 0
    failed_count = 0
    
    for file_path, original_stem, sanitized_stem in files_to_rename:
        # Construct new filename with sanitized stem but same extension
        new_filename = sanitized_stem + file_path.suffix
        new_path = file_path.parent / new_filename
        
        # Check if target file already exists
        if new_path.exists() and new_path != file_path:
            logger.warning(f"Skipping {file_path.name}: target file {new_filename} already exists")
            failed_count += 1
            continue
        
        if dry_run:
            logger.info(f"Would rename: {file_path.name} -> {new_filename}")
        else:
            try:
                file_path.rename(new_path)
                logger.info(f"Renamed: {file_path.name} -> {new_filename}")
                renamed_count += 1
            except OSError as e:
                logger.error(f"Failed to rename {file_path.name}: {e}")
                failed_count += 1
    
    if not dry_run:
        logger.info(f"Completed: {renamed_count} files renamed, {failed_count} failed")
    else:
        logger.info(f"Dry run completed: {len(files_to_rename)} files would be processed")

def main():
    parser = argparse.ArgumentParser(
        description="Rename files by sanitizing their filename stems using sanitize_filename function"
    )
    parser.add_argument(
        "directory",
        type=pathlib.Path,
        help="Directory containing files to rename"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process subdirectories recursively"
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually renaming files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(levelname)s: %(message)s'
    )
    
    try:
        rename_files(args.directory, args.recursive, args.dry_run)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
