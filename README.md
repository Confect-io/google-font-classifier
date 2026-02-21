# Finetuned DINOv2 Vision Transformer for categorizing Google Fonts


1. Get google fonts:

```
git clone --filter=blob:none --depth 1 https://github.com/google/fonts.git
```

2. Generate dataset:

```
python dataset_generator.py \
    --font_dir <wherever the Google fonts are downloaded to> \
    --out_dir <output folder> \
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
python dataset_cleaner.py <dataset folder>
```

This will print any bad image paths that you can manually inspect and remove (in expectation, only 1/225000 should be malformed).

4. Upload the dataset (optional):

You can upload the dataset to huggingface as follows:

```
huggingface-cli upload-large-folder <HuggingFace Username>/<Repo name> <path to folder> --repo-type=dataset
```

To get huggingface-cli, run

```
pip install -U "huggingface_hub[cli]"
```

5. Train the model:

To get the packages needed, make sure you have a venv (see above) and then get the packages:

```
python3 -m pip install transformers datasets torchvision peft accelerate bitsandbytes tensorboard
```

Then, train the model on the cleaned dataset ex:

```
python train_model.py \
    --data_dir <processed training + test data folder> \     
    --output_dir <where to output your model tensors to> \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
```

6. You can continue training where you left off.

Checkpoints and final model results will be saved in the output directory:

```
$ ls <output model dir>
checkpoint-2500 checkpoint-2752 logs
```

The outputs include the LoRA weights and classification head weights, so training from a checkpoint is equivalent to continuing training from the checkpoint state: 

```
python train_model.py \
    --checkpoint <output model dir>/checkpoint-2752 \
    ...
```


7. To upload your model, you can start from a checkpoint and train with 0 epochs i.e. skip training. This way the training code can inflate the checkpoint into model state. Then, pass the huggingface_model_name parameter.:

```
python train_model.py \
    --epochs 0
    --data_dir <input data dir> \
    --checkpoint <output model name>/checkpoint-2752 \
    --huggingface_model_name <your user name>/<your repo name>
```

You can modify the model repo via git:

```
git clone https://huggingface.co/dchen0/font-classifier
```

8. To serve the model from Huggingface, you can use the serve_model.py script on an image

```
python serve_model.py some_image.png
```


## Handling Expanded Datasets with Existing Checkpoints

If you've trained a model and later expanded your dataset with more font classes, you can continue training without starting from scratch using Automatic Size Mismatch Handling.

This method automatically detects size mismatches and loads compatible weights while initializing new classifier parameters randomly:

```bash
python train_model.py \
    --data_dir .data_out/PROD_font_dataset \
    --checkpoint path/to/your/checkpoint \
    --output_dir ./output \
    --epochs 5
```

The script will:
- Try to load the checkpoint normally
- If size mismatch is detected, fall back to loading only compatible weights
- Initialize new classifier parameters randomly
- Continue training from there


### Example: Expanding from 682 to 700 Classes

If your original checkpoint was trained on 682 font classes and you now have 700 classes, the regular checkpoint loading code will just work, no explicit configuration required:

```bash
python train_model.py \
    --data_dir .data_out/PROD_font_dataset \
    --checkpoint ./output/checkpoint-1500 \
    --output_dir ./continued_training \
    --epochs 3 \
    --learning_rate 1e-5  # Lower learning rate for fine-tuning
```

The script will show how many parameters are being loaded vs. initialized.

## Evaluation & Confusion Matrix

To evaluate the model on a test set and generate paper-ready figures:

```bash
python confusion_matrix.py \
    --data_dir <dataset folder> \
    --model <HuggingFace model name or local path>
```

**Important:** The model's label set must match the dataset's class folders. Use the model that was trained on (or matches) your dataset.

This produces:
- `figures/confusion_matrix.pdf` — Row-normalized confusion matrix heatmap grouped by font family
- `figures/top_confused_pairs.pdf` — Bar chart of the top-N most frequent misclassification pairs
- `confusion_matrix.json` — Full confusion matrix as a nested dict of counts
- `bad_images.json` — List of all misclassified images with true/predicted labels
- Per-class precision, recall, and F1 printed to stdout

Additional options:
- `--batch_size 32` — Batch size for inference (default: 32)
- `--top_n 20` — Number of confused pairs to plot (default: 20)
- `--output_dir figures` — Directory for output figures (default: figures)

Requires `scikit-learn` and `matplotlib` (listed in `requirements.txt`).

## Paper

Build the paper (runs evaluation then compiles LaTeX):

```bash
bash build_paper.sh
```

To skip the confusion matrix step and only compile LaTeX:

```bash
bash build_paper.sh --skip-matrix
```

Extra arguments are forwarded to `confusion_matrix.py`:

```bash
bash build_paper.sh --data_dir <dataset folder> --model <HuggingFace model name or local path>
```

## Handler.py

The handler.py module allows for the HF Inference endpoint to preprocess inbound images in the same way that images were processed for training and testing. Without this the model will be running on production data that does not match the format it expects.
