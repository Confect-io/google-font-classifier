#!/usr/bin/env python3
"""
Generate cropped glyph images for DINO‑v2 fine‑tuning.
"""
import argparse
import logging
import pathlib
import random
import sys

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

logger = logging.getLogger(__name__)
LENGTH_OF_STRINGS = 2

ASCII_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
)

FONT_ALLOWLIST = [
"Inter",
"Poppins",
"Arial",
"Roboto",
"InstrumentSans",
"InstrumentSerif",
"TimesNewRoman",
"Baskerville",
"PTSerif",
"LibreCaslonText",
"PermanentMarker",
"PinyonScript",
"Gluten",
"MeowScript",
"PatrickHand",
]

def font_is_variable(font_path: pathlib.Path) -> bool:
    return "fvar" in TTFont(str(font_path))

def char_set(name: str) -> str:
    if name == "ascii":
        return ASCII_CHARS
    if name == "letters":
        return ASCII_CHARS[:52]          # A‑Z a‑z only
    # Interpret any other string as a literal char list:
    return name

def render_and_crop(text: str, font: ImageFont.FreeTypeFont,
                     padding: int,
                    img_size: int) -> Image.Image:
    # Generate random background and text colors with sufficient contrast
    def generate_contrasting_colors():
        def random_rgb():
            return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        def luminance(color):
            # Calculate perceived brightness using standard formula
            r, g, b = color
            return 0.299 * r + 0.587 * g + 0.114 * b
        
        # Generate background color
        bg_color = random_rgb()
        bg_luminance = luminance(bg_color)
        
        # Generate text color with sufficient contrast
        min_luminance_diff = 80
        max_attempts = 50
        
        for _ in range(max_attempts):
            text_color = random_rgb()
            text_luminance = luminance(text_color)
            
            # Check if luminance difference is sufficient
            if abs(bg_luminance - text_luminance) >= min_luminance_diff:
                return bg_color, text_color
        
        # Fallback: if we can't find a good random contrast, use black/white
        if bg_luminance < 128:
            text_color = (255, 255, 255)  # white text on dark background
        else:
            text_color = (0, 0, 0)  # black text on light background
            
        return bg_color, text_color
    
    bg_color, text_color = generate_contrasting_colors()
    
    # Calculate text dimensions
    left, top, right, bottom = font.getbbox(text)
    text_width = right - left
    text_height = bottom - top
    
    # Calculate canvas dimensions with padding
    canvas_width = text_width + padding * 2
    canvas_height = text_height + padding * 2
    
    # Create canvas with random background color
    canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(canvas)
    
    # Calculate text position (centered)
    text_x = canvas_width // 2
    text_y = canvas_height // 2
    
    # Draw text with contrasting color
    draw.text(
        (text_x, text_y),
        text,
        fill=text_color,
        font=font,
        anchor="mm",
    )
    
    # Find bbox of non-background pixels
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
    resized_glyph = glyph.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Add gaussian noise using PIL's load/putpixel methods
    def add_gaussian_noise_pil(img, noise_factor=0.1):
        noise_std = noise_factor * 255
        
        # Create a copy to modify
        noisy_img = img.copy()
        pixels = noisy_img.load()
        
        width, height = img.size
        
        for x in range(width):
            for y in range(height):
                pixel = pixels[x, y]
                
                if isinstance(pixel, tuple):  # RGB image
                    noisy_pixel = tuple(
                        max(0, min(255, int(p + random.gauss(0, noise_std))))
                        for p in pixel
                    )
                else:  # Grayscale image
                    noisy_pixel = max(0, min(255, int(pixel + random.gauss(0, noise_std))))
                
                pixels[x, y] = noisy_pixel
        
        return noisy_img
    
    return add_gaussian_noise_pil(resized_glyph)


def build_dataset(font_dir, out_dir, chars, font_size, img_size, padding, no_clobber):
    font_dir, out_dir = pathlib.Path(font_dir), pathlib.Path(out_dir)
    train_dir, test_dir = out_dir / "train", out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    font_paths = list(font_dir.rglob("*.ttf")) + list(font_dir.rglob("*.otf"))
    font_paths = [font_path for font_path in font_paths if font_path.stem.split("[")[0].split("-")[0] in FONT_ALLOWLIST]
    if not font_paths:
        sys.exit(f"No font files found under {font_dir!s}")

    logger.info(f"Found {len(font_paths)} font files: {font_paths}")

    failed_fonts = []

    progress_bar = tqdm(font_paths, unit="font")
    for font_path in progress_bar:
        # font_path.stem is something like "Roboto-Regular" or "Roboto-Regular[wdth,wght]", we clip out the [wdth,wght] part (OpenType “variation axes”)
        font_family_name = font_path.stem.split("[")[0]
        progress_bar.set_description(font_family_name)

        try:
            font = ImageFont.truetype(str(font_path), font_size, layout_engine=ImageFont.Layout.BASIC)

            def generate_all_images_for_font(font: ImageFont.FreeTypeFont, font_name: str):
                font_train_dir = train_dir / font_name
                font_test_dir = test_dir / font_name
                font_train_dir.mkdir(exist_ok=True)
                font_test_dir.mkdir(exist_ok=True)

                strings_to_generate = []

                cur_frontier = [char for char in chars]
                strings_to_generate.extend(cur_frontier)

                for i in range(2,10):
                    for _ in range(100):
                        random_string = ''.join(random.choices(chars + ' ', k=i))
                        # skip all whitespace strings
                        if all(char in ' \n\t' for char in random_string):
                            continue
                        strings_to_generate.append(random_string)

                def generate_image_for_string(string: str, font: ImageFont.FreeTypeFont, root: pathlib.Path):
                    target_file = root / f"{font_name}_{string}.png"
                    if target_file.exists() and no_clobber:
                        logger.info(f"Skipping {target_file} because it already exists")
                        return
                    img = render_and_crop(string, font, padding, img_size)
                    if img is None:
                        logger.warning(f"Failed to render {string} for {font_name}")
                        return
                    img.save(target_file)
                    logger.info(f"Saved {target_file}")

                for string in strings_to_generate:
                    generate_image_for_string(string, font, font_train_dir)

                for i in range(2, 10):
                    random_string = ''.join(random.choices(chars + ' ', k=i))
                    generate_image_for_string(random_string, font, font_test_dir)

            
            if font_is_variable(font_path):
                font_variations = font.get_variation_names()
                for variation in font_variations:
                    font.set_variation_by_name(variation)
                    variation_str = variation.decode("utf-8").replace(" ", "_")
                    generate_all_images_for_font(font, f"{font_family_name}_{variation_str}")
            else:
                generate_all_images_for_font(font, font_family_name)


        except Exception as e:
            logger.error(f"{font_path.name}: {e}")
            failed_fonts.append(font_path)
            continue

    
    if failed_fonts:
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
