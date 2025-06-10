#!/usr/bin/env python3
"""
Generate cropped glyph images for DINO‑v2 fine‑tuning.
"""
import argparse, logging, pathlib, sys
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

def render_and_crop(char: str, font: ImageFont.FreeTypeFont,
                    render_size: int, padding: int,
                    img_size: int) -> Image.Image:
    # render large canvas
    side = render_size + padding * 2          # e.g. 1024 + 64
    centre = side // 2
    canvas = Image.new("L", (side, side), 255)  # white background
    draw   = ImageDraw.Draw(canvas)

    # Pillow ≥ 9.2 supports anchor="mm" (“middle, middle”)
    draw.text(
        (centre, centre),
        char,
        fill=0,
        font=font,
        anchor="mm",       # <‑‑ this centres the glyph
    )

    # find bbox of non‑white pixels
    bbox = canvas.getbbox()
    if not bbox:                         # empty glyph
        return None
    glyph = canvas.crop(bbox)

    # For strings, we want to maintain the aspect ratio but ensure the height fits
    # Calculate the target height while maintaining aspect ratio
    target_height = img_size
    aspect_ratio = glyph.width / glyph.height
    target_width = int(target_height * aspect_ratio)
    
    # Resize maintaining aspect ratio
    return glyph.resize((target_width, target_height), Image.LANCZOS)

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def build_dataset(font_dir, out_dir, chars, render_size, img_size, padding, no_clobber):
    font_dir, out_dir = pathlib.Path(font_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    font_paths = list(font_dir.rglob("*.ttf")) + list(font_dir.rglob("*.otf"))
    if not font_paths:
        sys.exit(f"No font files found under {font_dir!s}")

    failed_fonts = []

    progress_bar = tqdm(font_paths, unit="font")
    for font_path in progress_bar:
        progress_bar.set_description(font_path.stem)
        family_dir = out_dir / font_path.stem
        family_dir.mkdir(parents=True, exist_ok=True)

        try:
            font = ImageFont.truetype(str(font_path), render_size, layout_engine=ImageFont.Layout.BASIC)

            strings_to_generate = []

            cur_frontier = [char for char in chars]
            strings_to_generate.extend(cur_frontier)

            for _ in range(LENGTH_OF_STRINGS - 1):
                logger.info(f"Generating {len(cur_frontier)} strings")
                new_frontier = [cur_string + new_char for cur_string in cur_frontier for new_char in chars]
                strings_to_generate.extend(new_frontier)
                cur_frontier = new_frontier

            for string in strings_to_generate:
                target_file = family_dir / f"{font_path.stem}_{string}.png"
                if target_file.exists() and no_clobber:
                    logger.info(f"Skipping {target_file} because it already exists")
                    continue
                img = render_and_crop(string, font, render_size, padding, img_size)
                if img is None:
                    continue
                img.save(target_file)
                logging.info(f"Saved {target_file}")

        except Exception as e:
            failed_fonts.append(font_path)
            logging.error(f"⚠️  {font_path.name}: {e}")
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
    ap.add_argument("--render_size", type=int, default=1024,
                    help="Font size used for initial rendering")
    ap.add_argument("--padding",   type=int, default=64, help="Pixels of padding before crop")
    ap.add_argument("--no-clobber", action="store_true", help="Skip existing files, useful for rerunning when there are errors.")
    ap.add_argument("--verbose",   action="store_true", help="Verbose output")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    build_dataset(
        font_dir    = args.font_dir,
        out_dir     = args.out_dir,
        chars       = char_set(args.chars),
        render_size = args.render_size,
        img_size    = args.img_size,
        padding     = args.padding,
        no_clobber  = args.no_clobber,
    )

if __name__ == "__main__":
    cli()
