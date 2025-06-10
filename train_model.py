import os, torch, torchvision.transforms as T
import argparse
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    Dinov2ForImageClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DINOv2 model for font classification')
    parser.add_argument('--data_dir', type=str, default='fonts',
                      help='Directory containing the font dataset')
    parser.add_argument('--output_dir', type=str, default='dinov2-fonts',
                      help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training and evaluation batch size')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate for training')
    parser.add_argument('--lora_rank', type=int, default=8,
                      help='LoRA rank for parameter-efficient fine-tuning')
    parser.add_argument('--lora_alpha', type=int, default=16,
                      help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                      help='LoRA dropout rate')
    parser.add_argument('--test_size', type=float, default=0.1,
                      help='Proportion of data to use for validation')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--log_level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Configure logging with timestamps
    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    ######################################################################
    # 1. Load your cropped‑font dataset
    # -------------------------------------------------------------------
    # Directory layout expected by ImageFolder:
    #   fonts/
    #     ├─ Arial/
    #     │   ├─ img001.png
    #     │   └─ ...
    #     ├─ TimesNewRoman/
    #     └─ ...

    logger.info(f"Loading dataset from {args.data_dir}")
    dataset = load_dataset("imagefolder", data_dir=args.data_dir)
    
    # Debug: Print the first example to see its structure
    logger.info("Dataset structure:")
    logger.info(f"First example keys: {dataset['train'][0].keys()}")
    logger.info(f"First example: {dataset['train'][0]}")
    
    # Always perform train/test split
    logger.info(f"Performing train/test split with test_size={args.test_size}")
    dataset = dataset["train"].train_test_split(test_size=args.test_size, seed=args.seed)
    logger.info(f"Split complete. Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

    ######################################################################
    # 2. Pre‑processing & augmentation
    # -------------------------------------------------------------------
    logger.info("Setting up image processor and augmentations")
    processor   = AutoImageProcessor.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")  # 224 px
    size        = processor.size["shortest_edge"]            # 224 by default
    normalize   = T.Normalize(mean=processor.image_mean, std=processor.image_std)

    # Convert grayscale to RGB first
    to_rgb = T.Lambda(lambda img: img.convert('RGB'))

    train_aug   = T.Compose([
        to_rgb,  # Convert to RGB first
        T.RandomResizedCrop(size, scale=(0.9, 1.0)),
        T.RandomRotation(5),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
        T.ToTensor(), normalize,
    ])
    val_aug     = T.Compose([
        to_rgb,  # Convert to RGB first
        T.Resize(size + 32), 
        T.CenterCrop(size), 
        T.ToTensor(), 
        normalize
    ])

    def transform(example, train=True):
        # The dataset uses 'image' as the key for PIL images
        example["pixel_values"] = train_aug(example["image"]) if train else val_aug(example["image"])
        return example

    # Apply transformations to both splits
    logger.info("Applying data transformations")
    # Create new datasets with transformations
    train_dataset = dataset["train"].map(lambda x: transform(x, train=True), remove_columns=["image"])
    test_dataset = dataset["test"].map(lambda x: transform(x, train=False), remove_columns=["image"])
    logger.info("Data preprocessing complete")

    ######################################################################
    # 3. Load DINO v2 and (optionally) add LoRA adapters
    # -------------------------------------------------------------------
    logger.info("Loading DINOv2 model")
    num_labels = train_dataset.features["label"].num_classes
    logger.info(f"Number of classes: {num_labels}")
    
    # First load the model without the classification head
    model = Dinov2ForImageClassification.from_pretrained(
        "facebook/dinov2-base-imagenet1k-1-layer",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # Ignore the classification head size mismatch
    )

    # --- parameter‑efficient fine‑tune (comment out for full FT) ---
    logger.info("Configuring LoRA adapters")
    peft_cfg = LoraConfig(
        r             = args.lora_rank,          # rank
        lora_alpha    = args.lora_alpha,
        target_modules = ["query", "value"],  # Q & V proj in ViT blocks
        lora_dropout  = args.lora_dropout,
        bias          = "none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    ######################################################################
    # 4. Define Trainer
    # -------------------------------------------------------------------
    def collate(batch):
        # The transform function has already converted images to tensors and stored them in pixel_values
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    logger.info("Setting up training arguments")
    # Check if we're on MPS (Apple Silicon)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    training_args = TrainingArguments(
        output_dir          = args.output_dir,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        eval_steps                 = 500,  # Evaluate every 500 steps
        save_steps                = 500,  # Save checkpoint every 500 steps
        num_train_epochs          = args.epochs,
        learning_rate             = args.learning_rate,
        weight_decay              = 0.05,
        fp16                      = device.type == "cuda",  # Only use fp16 on CUDA devices
        save_total_limit          = 2,
        logging_dir               = os.path.join(args.output_dir, "logs"),
        logging_steps             = 10,
        report_to                 = "tensorboard",
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = test_dataset,
        data_collator   = collate,
    )

    ######################################################################
    # 5. Train & push to the Hub
    # -------------------------------------------------------------------
    logger.info("Starting training")
    trainer.train()
    logger.info("Training complete")
    # trainer.push_to_hub("your‑username/dinov2-font‑classifier")
