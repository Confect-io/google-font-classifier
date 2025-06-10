#!/usr/bin/env python3
"""
Generate cropped glyph images for DINO‑v2 fine‑tuning.
"""
import argparse
import logging
import pathlib
import random
import sys

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

logger = logging.getLogger(__name__)
LENGTH_OF_STRINGS = 2

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
ASCII_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
)

def char_set(name: str) -> str:
    if name == "ascii":
        return ASCII_CHARS
    if name == "letters":
        return ASCII_CHARS[:52]          # A‑Z a‑z only
    # Interpret any other string as a literal char list:
    return name

def render_and_crop(text: str, font: ImageFont.FreeTypeFont,
                    font_size: int, padding: int,
                    img_size: int) -> Image.Image:
    # Calculate text dimensions
    left, top, right, bottom = font.getbbox(text)
    text_width = right - left
    text_height = bottom - top
    
    # Calculate canvas dimensions with padding
    canvas_width = text_width + padding * 2
    canvas_height = text_height + padding * 2
    
    # Create canvas with white background
    canvas = Image.new("L", (canvas_width, canvas_height), 255)
    draw = ImageDraw.Draw(canvas)
    
    # Calculate text position (centered)
    text_x = canvas_width // 2
    text_y = canvas_height // 2
    
    # Draw text
    draw.text(
        (text_x, text_y),
        text,
        fill=0,
        font=font,
        anchor="mm",
    )
    
    # Find bbox of non-white pixels
    bbox = canvas.getbbox()
    if not bbox:  # empty glyph
        return None
    glyph = canvas.crop(bbox)
    
    # For text sequences, we want to maintain the aspect ratio but ensure the height fits
    # Calculate the target height while maintaining aspect ratio
    target_height = img_size
    aspect_ratio = glyph.width / glyph.height
    target_width = int(target_height * aspect_ratio)
    
    # Resize maintaining aspect ratio
    return glyph.resize((target_width, target_height), Image.Resampling.LANCZOS)

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def build_dataset(font_dir, out_dir, chars, font_size, img_size, padding, no_clobber):
    font_dir, out_dir = pathlib.Path(font_dir), pathlib.Path(out_dir)
    train_dir, test_dir = out_dir / "train", out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    font_paths = list(font_dir.rglob("*.ttf")) + list(font_dir.rglob("*.otf"))
    if not font_paths:
        sys.exit(f"No font files found under {font_dir!s}")

    failed_fonts = []

    progress_bar = tqdm(font_paths, unit="font")
    for font_path in progress_bar:
        progress_bar.set_description(font_path.stem)
        family_train_dir = train_dir / font_path.stem
        family_test_dir = test_dir / font_path.stem
        family_train_dir.mkdir(parents=True, exist_ok=True)
        family_test_dir.mkdir(parents=True, exist_ok=True)

        try:
            font = ImageFont.truetype(str(font_path), font_size, layout_engine=ImageFont.Layout.BASIC)

            strings_to_generate = []

            cur_frontier = [char for char in chars]
            strings_to_generate.extend(cur_frontier)

            for i in range(2,10):
                random_string = ''.join(random.choices(chars + ' ', k=i))
                strings_to_generate.append(random_string)

            def generate_image_for_string(string: str, font: ImageFont.FreeTypeFont, root: pathlib.Path):
                target_file = root / f"{font_path.stem}_{string}.png"
                if target_file.exists() and no_clobber:
                    logger.info(f"Skipping {target_file} because it already exists")
                    return
                img = render_and_crop(string, font, font_size, padding, img_size)
                if img is None:
                    logger.warning(f"Failed to render {string} for {font_path.stem}")
                    return
                img.save(target_file)
                logging.info(f"Saved {target_file}")

            for string in strings_to_generate:
                generate_image_for_string(string, font, family_train_dir)

            for i in range(2, 10):
                random_string = ''.join(random.choices(chars + ' ', k=i))
                generate_image_for_string(random_string, font, family_test_dir)

        except Exception as e:
            failed_fonts.append(font_path)
            logging.error(f"{font_path.name}: {e}")
            continue

        logging.info(f"Processed {len(chars)} glyphs for {font_path.stem}")
    logging.warning(f"Failed to process {len(failed_fonts)} fonts: {failed_fonts}")

def cli():
    ap = argparse.ArgumentParser(description="Crop glyphs for DINO v2 fine‑tuning")
    ap.add_argument("--font_dir",  required=True, help="Directory with TTF/OTF files")
    ap.add_argument("--out_dir",   default="glyphs224", help="Destination root folder")
    ap.add_argument("--chars",     default="ascii",
                    help="'ascii', 'letters', or a literal string of chars")
    ap.add_argument("--img_size",  type=int, default=224, help="Final square size (px)")
    ap.add_argument("--font_size", type=int, default=1024,
                    help="Font size used for initial rendering")
    ap.add_argument("--padding",   type=int, default=128, help="Pixels of padding before crop")
    ap.add_argument("--no-clobber", action="store_true", help="Skip existing files, useful for rerunning when there are errors.")
    ap.add_argument("--verbose",   action="store_true", help="Verbose output")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    build_dataset(
        font_dir    = args.font_dir,
        out_dir     = args.out_dir,
        chars       = char_set(args.chars),
        font_size   = args.font_size,
        img_size    = args.img_size,
        padding     = args.padding,
        no_clobber  = args.no_clobber,
    )

if __name__ == "__main__":
    cli()
