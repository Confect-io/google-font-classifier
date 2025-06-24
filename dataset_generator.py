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
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    " \n\t"
)

FONT_ALLOWLIST = [
"Lato",
"AnkeDevanagari",
"Merriweather",
"Alegreya",
"Montserrat",
"Aleo",
"Muli",
"Arapey",
"Nunito",
"AsapCondensed",
"Assistant",
"OpenSans",
"Barlow",
"Oswald",
"Bitter",
"Poppins",
"Brawler",
"Roboto",
"Caladea",
"ROKKITT",
"Carme",
"Rubik",
"EncodeSansSemiCondensed",
"Enriqueta",
"SourceSans3",
"FrankRuhlLibre",
"Spectral",
"WorkSans",
"Gelasio",
"HeadlandOne",
"Lora",
"CrimsonText",
"PlayfairDisplay",
"PTSerif",
"Raleway",
"SourceCodePro",
"Ubuntu",
"RobotoCondensed",
"JosefinSans",
"Cabin",
"Domine",
"FiraSans",
"Inconsolata",
"Karla",
"LibreBaskerville",
"Maitree",
"NanumGothic",
"Quattrocento",
"Teko",
"ZillaSlab",
"Inter",
"InstrumentSerif",
"InstrumentSans",
"CedarvillCursive",
"Collapse",
]

def font_is_variable(font_path: pathlib.Path) -> bool:
    return "fvar" in TTFont(str(font_path))

def char_set(name: str) -> str:
    if name == "ascii":
        return ASCII_CHARS
    if name == "letters":
        return ASCII_CHARS[:52] + " \n\t"          # A‑Z a‑z only
    # Interpret any other string as a literal char list:
    return name

def sanitize_filename(text: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    # Replace problematic characters with safe alternatives
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
        ' ': '_space_'
    }
    
    sanitized = text
    for char, replacement in replacements.items():
        sanitized = sanitized.replace(char, replacement)

    if len(sanitized) > 200:
        sanitized = "nameTooLong" + str(random.randint(0, 1000000))
    
    return sanitized

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
    
    # Handle multi-line text properly
    lines = text.split('\n')
    
    # Calculate dimensions for multi-line text
    line_height = font.getbbox('Ay')[3] - font.getbbox('Ay')[1]  # Height from A to y (covers ascenders and descenders)
    line_spacing = int(line_height * 0.2)  # 20% additional spacing between lines
    
    # Calculate total text dimensions
    max_width = 0
    total_height = len(lines) * line_height + (len(lines) - 1) * line_spacing
    
    for line in lines:
        if line.strip():  # Skip empty lines for width calculation
            left, top, right, bottom = font.getbbox(line)
            line_width = right - left
            max_width = max(max_width, line_width)
    
    # Use larger dimensions for canvas to ensure nothing gets cropped
    canvas_width = max_width + padding * 2
    canvas_height = int(total_height) + padding * 2
    
    # Create canvas with random background color
    canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(canvas)
    
    # Draw each line of text with random alignment
    alignment = random.choice(['left', 'center', 'right'])
    start_y = padding
    for i, line in enumerate(lines):
        if line.strip():  # Skip completely empty lines
            line_bbox = font.getbbox(line)
            line_width = line_bbox[2] - line_bbox[0]
            
            # Calculate x position based on random alignment
            if alignment == 'left':
                text_x = padding
            elif alignment == 'center':
                text_x = (canvas_width - line_width) // 2
            else:  # right
                text_x = canvas_width - line_width - padding
                
            text_y = start_y + i * (line_height + line_spacing)
            
            draw.text(
                (text_x, text_y),
                line,
                fill=text_color,
                font=font,
                anchor="lt",  # left-top anchor for precise positioning
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


def choose_sentence():
    """Choose a random substring from text files in input_data directory."""
    
    input_data_dir = pathlib.Path("input_data")
    if not input_data_dir.exists():
        return None
    
    # Find all text files
    text_files = list(input_data_dir.glob("*.txt"))
    if not text_files:
        return None
    
    # Choose a random text file
    text_file = random.choice(text_files)
    
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if len(content) < 10:  # Skip very short files
            return None
        
        # Choose random substring length (between 5 and 100 characters)
        max_length = min(200, len(content))
        substring_length = random.randint(20, max_length)
        
        # Choose random starting position
        start_pos = random.randint(0, len(content) - substring_length)
        
        # Extract substring
        substring = content[start_pos:start_pos + substring_length]

        # Randomly replace some spaces with newlines
        CHANCE_TO_REPLACE_SPACE_WITH_NEWLINE = 0.2
        substring = ''.join('\n' if char == ' ' and random.random() < CHANCE_TO_REPLACE_SPACE_WITH_NEWLINE else char for char in substring)
        
        return substring.strip()
        
    except Exception as e:
        logger.warning(f"Error reading {text_file}: {e}")
        return None



def build_dataset(font_dir, out_dir, chars, font_size, img_size, padding, no_clobber):
    font_dir, out_dir = pathlib.Path(font_dir), pathlib.Path(out_dir)
    train_dir, test_dir = out_dir / "train", out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    font_paths = list(font_dir.rglob("*.ttf")) + list(font_dir.rglob("*.otf"))
    font_paths = [font_path for font_path in font_paths if font_path.stem.split("[")[0].split("-")[0] in FONT_ALLOWLIST]
    if not font_paths:
        sys.exit(f"No font files found under {font_dir!s}")

    if len(font_paths) < len(FONT_ALLOWLIST):
        missing_fonts = [font for font in FONT_ALLOWLIST if not any(font in font_path for font_path in font_paths)]
        raise ValueError(f"Not enough font files found under {font_dir!s}: {missing_fonts}")

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
                def generate_image_for_string(string: str, font: ImageFont.FreeTypeFont, root: pathlib.Path):
                    # Sanitize the string for use in filename
                    safe_string = sanitize_filename(string)
                    target_file = root / f"{font_name}_{safe_string}.png"
                    if target_file.exists() and no_clobber:
                        logger.info(f"Skipping {target_file} because it already exists")
                        return
                    img = render_and_crop(string, font, padding, img_size)
                    if img is None:
                        logger.warning(f"Failed to render {string} for {font_name}")
                        return
                    logger.info(f"Saving {target_file}")
                    img.save(target_file)
                    logger.info(f"Saved {target_file}")
                    
                font_train_dir = train_dir / font_name
                font_test_dir = test_dir / font_name
                font_train_dir.mkdir(exist_ok=True)
                font_test_dir.mkdir(exist_ok=True)

                # Training set
                strings_to_generate = [char for char in chars if char not in ['\n', '\t', ' ']]

                for i in range(2,100):
                    for _ in range(10):
                        random_string = ''.join(random.choices(chars, k=i))
                        # skip all whitespace strings
                        if all(char in ' \n\t' for char in random_string):
                            continue
                        strings_to_generate.append(random_string)
                
                # Add random sentences from input_data
                for _ in range(500):  # Generate 500 random sentences
                    sentence = choose_sentence()
                    if sentence:
                        strings_to_generate.append(sentence)

                for string in strings_to_generate:
                    generate_image_for_string(string, font, font_train_dir)

                # Test set
                for i in range(2, 100):
                    for _ in range(10):
                        random_string = ''.join(random.choices(chars, k=i))
                        if all(char in ['\n', '\t', ' '] for char in random_string):
                            continue
                        generate_image_for_string(random_string, font, font_test_dir)
                
                # Add random sentences to test set
                for _ in range(50):  # Generate 50 random sentences for test
                    sentence = choose_sentence()
                    if sentence:
                        generate_image_for_string(sentence, font, font_test_dir)

            
            if font_is_variable(font_path):
                font_variations = font.get_variation_names()
                for variation in font_variations:
                    font.set_variation_by_name(variation)
                    variation_str = variation.decode("utf-8").replace(" ", "_")
                    generate_all_images_for_font(font, f"{font_family_name}_{variation_str}")
            else:
                generate_all_images_for_font(font, font_family_name)


        except Exception as e:
            import traceback
            logger.error(f"Failed to process font {font_path.name}:")
            logger.error(f"  Error: {e}")
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            failed_fonts.append((font_path, str(e)))
            continue

    
    if failed_fonts:
        logger.warning(f"Failed to process {len(failed_fonts)} fonts:")
        for font_path, error in failed_fonts:
            logger.warning(f"  {font_path.name}: {error}")

def cli():
    ap = argparse.ArgumentParser(description="Crop glyphs for DINO v2 fine‑tuning")
    ap.add_argument("--font_dir",  required=True, help="Directory with TTF/OTF files")
    ap.add_argument("--out_dir",   default="glyphs224", help="Destination root folder")
    ap.add_argument("--chars",     default="ascii",
                    help="'ascii', 'letters', or a literal string of chars")
    ap.add_argument("--img_size",  type=int, default=224, help="Final square size (px)")
    ap.add_argument("--font_size", type=int, default=1024,
                    help="Font size used for initial rendering")
    ap.add_argument("--padding",   type=int, default=500, help="Pixels of padding before crop")
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
