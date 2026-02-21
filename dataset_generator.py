#!/usr/bin/env python3
"""
Generate cropped glyph images for DINO‑v2 fine‑tuning.
"""
import argparse
import logging
import multiprocessing
import os
import pathlib
import random
import sys

import numpy as np

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Disable PIL decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)

ASCII_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    " \n\t"
)

FONT_ALLOWLIST = [
"BigShouldersText",
"BricolageGrotesque",
"CrimsonPro",
"DMSans",
"Geist",
"HedvigLettersSerif",
"InstrumentSans",
"InstrumentSerif",
"JetBrainsMono",
"LexendDeca",
"Lora",
"Montserrat",
"Newsreader",
"NunitoSans",
"Onest",
"Petrona",
"PlayfairDisplay",
"PlusJakartaSans",
"Poppins",
"PT_Serif_Caption",
"RethinkSans",
"RobotoSerif",
"ShipporiMincho",
"Sora",
"SpaceGrotesk",
"Ultra",
"Urbanist",
"Inter",
"WorkSans",
"Merriweather",
"OpenSans",
"Roboto",
]

# ---------------------------------------------------------------------------
# Text corpus (preloaded into memory for workers)
# ---------------------------------------------------------------------------

_TEXT_CORPUS = None  # populated by _load_text_corpus()

def _load_text_corpus():
    """Read all text files from input_data/ into a list of strings."""
    input_data_dir = pathlib.Path("input_data")
    if not input_data_dir.exists():
        raise ValueError(f"Input data directory {input_data_dir} does not exist")
    texts = []
    for txt_file in sorted(input_data_dir.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8").strip()
        if len(content) >= 100:
            texts.append(content)
    if not texts:
        raise ValueError(f"No usable text files found in {input_data_dir}")
    return texts


def choose_sentence(corpus):
    """Choose a random substring from the preloaded corpus."""
    content = random.choice(corpus)
    substring_length = random.randint(20, 100)
    start_pos = random.randint(0, len(content) - substring_length)
    substring = content[start_pos:start_pos + substring_length]
    # Randomly replace some spaces with newlines
    substring = ''.join(
        '\n' if c == ' ' and random.random() < 0.2 else c
        for c in substring
    )
    return substring.strip() or None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def font_is_variable(font_path: pathlib.Path) -> bool:
    return "fvar" in TTFont(str(font_path))

def char_set(name: str) -> str:
    if name == "ascii":
        return ASCII_CHARS
    if name == "letters":
        return ASCII_CHARS[:52] + " \n\t"
    return name

def sanitize_filename(text: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    replacements = {
        '/': '_slash_',
        '\\': '_backslash_',
        ':': '_colon_',
        '*': '_star_',
        '?': '_question_',
        '"': '_quote_',
        '<': '_lt_',
        '>': '_gt_',
        '|': '_pipe_',
        '\n': '_newline_',
        '\t': '_tab_',
        ' ': '_space_',
        '`': '_backtick_',
        '~': '_tilde_',
        '!': '_exclamation_',
        '@': '_at_',
        '#': '_hash_',
        '$': '_dollar_',
        '%': '_percent_',
        '^': '_caret_',
        '&': '_ampersand_',
        '(': '_lparen_',
        ')': '_rparen_',
        '{': '_lbrace_',
        '}': '_rbrace_',
        '[': '_lbracket_',
        ']': '_rbracket_',
        ';': '_semicolon_',
        ',': '_comma_',
        '.': '_dot_',
        "'": "_single_quote_",
    }
    sanitized = text
    for char, replacement in replacements.items():
        sanitized = sanitized.replace(char, replacement)
    if len(sanitized) > 200:
        sanitized = "nameTooLong" + str(random.randint(0, 1000000))
    return sanitized


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_and_crop(text, font, padding, img_size):
    # Generate random background and text colors with sufficient contrast
    def random_rgb():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def luminance(color):
        r, g, b = color
        return 0.299 * r + 0.587 * g + 0.114 * b

    bg_color = random_rgb()
    bg_lum = luminance(bg_color)
    text_color = None
    for _ in range(50):
        candidate = random_rgb()
        if abs(bg_lum - luminance(candidate)) >= 80:
            text_color = candidate
            break
    if text_color is None:
        text_color = (255, 255, 255) if bg_lum < 128 else (0, 0, 0)

    lines = text.split('\n')
    line_height = font.getbbox('Ay')[3] - font.getbbox('Ay')[1]
    line_spacing = int(line_height * 0.2)

    # Word-wrap long lines to keep the aspect ratio reasonable.
    # Without this, a long single-line sentence renders as e.g. 8000x224,
    # which after pad-to-square + resize becomes a tiny unreadable stripe.
    # Cap each line at ~8 * line_height pixels wide (produces ~2:1 to 4:1 images).
    max_line_px = line_height * 8
    wrapped = []
    for line in lines:
        words = line.split(' ')
        current = words[0] if words else ''
        for word in words[1:]:
            test = current + ' ' + word
            bbox = font.getbbox(test)
            if bbox[2] - bbox[0] > max_line_px:
                wrapped.append(current)
                current = word
            else:
                current = test
        wrapped.append(current)
    lines = wrapped

    total_height = len(lines) * line_height + (len(lines) - 1) * line_spacing

    max_width = 0
    for line in lines:
        if line.strip():
            left, top, right, bottom = font.getbbox(line)
            max_width = max(max_width, right - left)

    canvas_width = max_width + padding * 2
    canvas_height = int(total_height) + padding * 2

    canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    alignment = random.choice(['left', 'center', 'right'])
    start_y = padding
    for i, line in enumerate(lines):
        if line.strip():
            line_bbox = font.getbbox(line)
            line_width = line_bbox[2] - line_bbox[0]
            if alignment == 'left':
                text_x = padding
            elif alignment == 'center':
                text_x = (canvas_width - line_width) // 2
            else:
                text_x = canvas_width - line_width - padding
            text_y = start_y + i * (line_height + line_spacing)
            draw.text((text_x, text_y), line, fill=text_color, font=font, anchor="lt")

    bbox = canvas.getbbox()
    if not bbox:
        return None
    glyph = canvas.crop(bbox)

    target_height = img_size
    aspect_ratio = glyph.width / glyph.height
    target_width = int(target_height * aspect_ratio)
    resized_glyph = glyph.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Add gaussian noise (vectorized)
    arr = np.array(resized_glyph, dtype=np.float32)
    noise = np.random.normal(0, 0.1 * 255, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Per-variant worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def _worker_init(corpus):
    """Store corpus in each worker's global state."""
    global _TEXT_CORPUS
    _TEXT_CORPUS = corpus


def _generate_variant(args):
    """Generate all images for one font variant. Designed for multiprocessing."""
    font_path, font_name, variation_name, train_dir, test_dir, font_size, img_size, padding, no_clobber = args

    font = ImageFont.truetype(str(font_path), font_size, layout_engine=ImageFont.Layout.BASIC)
    if variation_name is not None:
        font.set_variation_by_name(variation_name)
        variant_str = variation_name.decode("utf-8").replace(" ", "_")
        full_name = f"{font_name}_{variant_str}"
    else:
        full_name = font_name

    font_train_dir = pathlib.Path(train_dir) / full_name
    font_test_dir = pathlib.Path(test_dir) / full_name
    font_train_dir.mkdir(parents=True, exist_ok=True)
    font_test_dir.mkdir(parents=True, exist_ok=True)

    corpus = _TEXT_CORPUS

    def generate_image(string, root):
        safe_string = sanitize_filename(string)
        target_file = root / f"{full_name}_{safe_string}.png"
        if target_file.exists() and no_clobber:
            return
        img = render_and_crop(string, font, padding, img_size)
        if img is not None:
            img.save(target_file, compress_level=1)

    # Training set
    for _ in range(500):
        sentence = choose_sentence(corpus)
        if sentence:
            generate_image(sentence, font_train_dir)
    for _ in range(25):
        generate_image(f"{random.randint(1, 1000000)}", font_train_dir)
    for _ in range(25):
        generate_image(f"${random.randint(1, 1000000)}", font_train_dir)
    for _ in range(25):
        generate_image(f"{random.randint(0, 100)}%", font_train_dir)

    # Test set
    for _ in range(25):
        sentence = choose_sentence(corpus)
        if sentence:
            generate_image(sentence, font_test_dir)
    for _ in range(5):
        generate_image(f"{random.randint(1, 1000000)}", font_test_dir)
    for _ in range(5):
        generate_image(f"${random.randint(1, 1000000)}", font_test_dir)
    for _ in range(5):
        generate_image(f"{random.randint(0, 100)}%", font_test_dir)

    return full_name


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(font_dir, out_dir, chars, font_size, img_size, padding, no_clobber, workers):
    font_dir, out_dir = pathlib.Path(font_dir), pathlib.Path(out_dir)
    train_dir, test_dir = out_dir / "train", out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    font_paths = list(font_dir.rglob("*.ttf")) + list(font_dir.rglob("*.otf"))
    allowlist_lower = [f.lower() for f in FONT_ALLOWLIST]
    font_paths = [fp for fp in font_paths if fp.stem.split("[")[0].split("-")[0].lower() in allowlist_lower]
    if not font_paths:
        sys.exit(f"No font files found under {font_dir!s}")

    missing_fonts = [f for f in FONT_ALLOWLIST if not any(f.lower() in fp.stem.lower() for fp in font_paths)]
    if missing_fonts:
        raise ValueError(f"Missing fonts under {font_dir!s}: {missing_fonts}")

    # Preload text corpus
    corpus = _load_text_corpus()

    # Enumerate all (font_path, variant) work items, deduplicating by output label
    work_items = []
    seen_labels = {}  # label -> font_path (for duplicate detection)
    for font_path in sorted(font_paths):
        font_family_name = font_path.stem.split("[")[0]
        try:
            if font_is_variable(font_path):
                font = ImageFont.truetype(str(font_path), font_size, layout_engine=ImageFont.Layout.BASIC)
                for variation in font.get_variation_names():
                    variant_str = variation.decode("utf-8").replace(" ", "_")
                    label = f"{font_family_name}_{variant_str}"
                    if label in seen_labels:
                        logger.warning(f"Skipping duplicate label '{label}' from {font_path.name} (already from {seen_labels[label].name})")
                        continue
                    seen_labels[label] = font_path
                    work_items.append((
                        font_path, font_family_name, variation,
                        str(train_dir), str(test_dir),
                        font_size, img_size, padding, no_clobber,
                    ))
            else:
                label = font_family_name
                if label in seen_labels:
                    logger.warning(f"Skipping duplicate label '{label}' from {font_path.name} (already from {seen_labels[label].name})")
                    continue
                seen_labels[label] = font_path
                work_items.append((
                    font_path, font_family_name, None,
                    str(train_dir), str(test_dir),
                    font_size, img_size, padding, no_clobber,
                ))
        except Exception as e:
            logger.error(f"Failed to enumerate variants for {font_path.name}: {e}")

    print(f"Found {len(work_items)} unique font variants from {len(font_paths)} font files")
    print(f"Generating images using {workers} workers ...")

    with multiprocessing.Pool(workers, initializer=_worker_init, initargs=(corpus,)) as pool:
        for name in tqdm(
            pool.imap_unordered(_generate_variant, work_items),
            total=len(work_items),
            unit="variant",
        ):
            pass

    # --- Post-generation validation ---
    train_variants = sorted(d for d in os.listdir(train_dir) if os.path.isdir(train_dir / d))
    test_variants = sorted(d for d in os.listdir(test_dir) if os.path.isdir(test_dir / d))

    train_total = sum(
        len([f for f in os.listdir(train_dir / v) if f.endswith(".png")])
        for v in train_variants
    )
    test_total = sum(
        len([f for f in os.listdir(test_dir / v) if f.endswith(".png")])
        for v in test_variants
    )

    print(f"\n--- Generation summary ---")
    print(f"  Output:          {out_dir}")
    print(f"  Train variants:  {len(train_variants)}")
    print(f"  Test variants:   {len(test_variants)}")
    print(f"  Train images:    {train_total} ({train_total / max(len(train_variants), 1):.0f} per variant)")
    print(f"  Test images:     {test_total} ({test_total / max(len(test_variants), 1):.0f} per variant)")

    if set(train_variants) != set(test_variants):
        missing_test = set(train_variants) - set(test_variants)
        missing_train = set(test_variants) - set(train_variants)
        if missing_test:
            print(f"  WARNING: {len(missing_test)} variants in train but not test: {sorted(missing_test)[:5]}")
        if missing_train:
            print(f"  WARNING: {len(missing_train)} variants in test but not train: {sorted(missing_train)[:5]}")

    # Check for empty variant dirs
    empty_train = [v for v in train_variants if len(os.listdir(train_dir / v)) == 0]
    empty_test = [v for v in test_variants if len(os.listdir(test_dir / v)) == 0]
    if empty_train:
        print(f"  WARNING: {len(empty_train)} empty train dirs: {empty_train[:5]}")
    if empty_test:
        print(f"  WARNING: {len(empty_test)} empty test dirs: {empty_test[:5]}")

    print(f"Done.")


def cli():
    ap = argparse.ArgumentParser(description="Crop glyphs for DINO v2 fine‑tuning")
    ap.add_argument("--font_dir",  required=True, help="Directory with TTF/OTF files")
    ap.add_argument("--out_dir",   default="glyphs224", help="Destination root folder")
    ap.add_argument("--chars",     default="ascii",
                    help="'ascii', 'letters', or a literal string of chars")
    ap.add_argument("--img_size",  type=int, default=224, help="Final square size (px)")
    ap.add_argument("--font_size", type=int, default=48,
                    help="Font size used for initial rendering")
    ap.add_argument("--padding",   type=int, default=50, help="Pixels of padding before crop")
    ap.add_argument("--no-clobber", action="store_true", help="Skip existing files, useful for rerunning when there are errors.")
    ap.add_argument("--workers",   type=int, default=None,
                    help="Number of parallel workers (default: number of CPU cores)")
    ap.add_argument("--verbose",   action="store_true", help="Verbose output")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    workers = args.workers or os.cpu_count() or 1

    build_dataset(
        font_dir    = args.font_dir,
        out_dir     = args.out_dir,
        chars       = char_set(args.chars),
        font_size   = args.font_size,
        img_size    = args.img_size,
        padding     = args.padding,
        no_clobber  = args.no_clobber,
        workers     = workers,
    )

if __name__ == "__main__":
    cli()
