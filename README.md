# Finetuned DINOv2 Vision Transformer for categorizing Google Fonts


1. Get google fonts:

```
git clone --filter=blob:none --depth 1 https://github.com/google/fonts.git
```

2. Generate dataset:

```
python dataset_generator.py \
    --font_dir ../fonts/ofl \
    --out_dir ./glyphs224 \
    --chars ascii \
    --img_size 224 \
    --render_size 1024 \
    --padding 32
```

To get the packages needed, create a venv

```
python3 -m venv myEnv
source myEnv/bin/activate
```

Then, install the packages

```
python3 -m pip install tqdm
python3 -m pip install pillow
```
