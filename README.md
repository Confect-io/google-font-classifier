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
    --font_size 1024 \
    --padding 128
```

To get the packages needed, create a venv

```
python3 -m venv myEnv
source myEnv/bin/activate
```

Then, install the packages

```
python3 -m pip install tqdm pillow fontTools
```

3. Clean the dataset:
```
python dataset_cleaner.py glyphs224
```

This will print any bad image paths that you can manually inspect and remove (in expectation, only 1/225000 should be malformed).

4. Train the model:

To get the packages needed, make sure you have a venv (see above) and then get the packages:

```
python3 -m pip install transformers datasets torchvision peft accelerate bitsandbytes tensorboard
```

Then, train the model on the cleaned dataset ex:

```
python train_model.py \
    --data_dir glyphs224 \     
    --output_dir dinov2-fonts \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
```

5. Checkpoints and final model results will be saved in the output directory:

```
$ ls dinov2-fonts-with-subfonts
checkpoint-2500 checkpoint-2752 logs
```

The outputs include the LoRA weights and classification head weights, so training from a checkpoint is equivalent to continuing training from the checkpoint state: 

```
python train_model.py \
    --checkpoint dinov2-fonts-with-subfonts/checkpoint-2752 \
    ...
```
