import argparse
import json
import logging
import os

import torch
import torchvision.transforms as T
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from transformers import (
    AutoImageProcessor,
    Dinov2ForImageClassification,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DINOv2 model for font classification')
    parser.add_argument('--data_dir', type=str, default='fonts',
                      help='Directory containing the font dataset')
    parser.add_argument('--output_dir', type=str, default='dinov2-fonts',
                      help='Directory to save the model')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint to resume training from')
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
    # Get label names from directory names
    label_names = os.listdir(f"{args.data_dir}/train")
    logger.info(f"Found {len(label_names)} labels")

    if len(label_names) <= 1:
        raise ValueError(f"Expected at least 2 labels, got {label_names=}, imagefolder will not label the dataset if there are less than 2 labels.")

    dataset = load_dataset(
        "imagefolder",
        data_dir=args.data_dir,
    )
    
    logger.info(f"Train size: {len(dataset['train'])}, Validation size: {len(dataset['test'])}")

    ######################################################################
    # 2. Pre‑processing & augmentation
    # -------------------------------------------------------------------
    logger.info("Setting up image processor and augmentations")
    processor   = AutoImageProcessor.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")  # 224 px
    size        = processor.size["shortest_edge"]            # 224 by default
    normalize   = T.Normalize(mean=processor.image_mean, std=processor.image_std)

    # Convert grayscale to RGB first
    to_rgb = T.Lambda(lambda img: img.convert('RGB'))
 
    # Define padding transform to ensure square images
    def pad_to_square(img):
        w, h = img.size
        max_size = max(w, h)
        pad_w = (max_size - w) // 2
        pad_h = (max_size - h) // 2
        padding = (pad_w, pad_h, max_size - w - pad_w, max_size - h - pad_h)
        return T.Pad(padding, fill=0)(img)

    train_aug   = T.Compose([
        to_rgb,  # Convert to RGB first
        pad_to_square,  # Pad to square
        T.Resize(size),  # Resize to target size
        T.ToTensor(), 
        normalize,
    ])
    val_aug     = T.Compose([
        to_rgb,  # Convert to RGB first
        pad_to_square,  # Pad to square
        T.Resize(size),  # Resize to target size
        T.ToTensor(), 
        normalize
    ])

    def transform(example, train=True):
        # The dataset uses 'image' as the key for PIL images
        example["pixel_values"] = train_aug(example["image"]) if train else val_aug(example["image"])
        # The label is already set by the ImageFolder dataset loader
        return example

    # Apply transformations to all splits
    logger.info("Applying data transformations")
    # Create new datasets with transformations
    train_dataset = dataset["train"].map(
        lambda x: transform(x, train=True),
        remove_columns=["image"],
        desc="Transforming training data"
    )
    test_dataset = dataset["test"].map(
        lambda x: transform(x, train=False),
        remove_columns=["image"],
        desc="Transforming test data"
    )
    
    # Set the format to torch tensors
    train_dataset.set_format(type="torch", columns=["pixel_values", "label"])
    test_dataset.set_format(type="torch", columns=["pixel_values", "label"])
    
    logger.info("Data preprocessing complete")

    ######################################################################
    # 3. Load DINO v2 and (optionally) add LoRA adapters
    # -------------------------------------------------------------------
    logger.info("Loading DINOv2 model")
    
    # First load the model without the classification head
    model = Dinov2ForImageClassification.from_pretrained(
            "facebook/dinov2-base-imagenet1k-1-layer",
            num_labels=len(label_names),
            ignore_mismatched_sizes=True,
        )

    # --- parameter‑efficient fine‑tune (comment out for full FT) ---
    logger.info("Configuring LoRA adapters")
    peft_cfg = LoraConfig(
        r             = args.lora_rank,          # rank
        lora_alpha    = args.lora_alpha,
        target_modules = ["query", "value"],  # Q & V proj in ViT blocks
        lora_dropout  = args.lora_dropout,
        bias          = "none",
        modules_to_save = ["classifier"],  # IMPORTANT: Save classification head too!
    )

    # Handle checkpoint loading properly
    checkpoint_for_trainer = None
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
                
        # Check if it's a PEFT checkpoint with adapter weights
        adapter_path = os.path.join(args.checkpoint, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"PEFT checkpoint not found at {adapter_path}")
        
        logger.info("Found PEFT checkpoint...")
        
        # Check if we have enhanced checkpoint with full model state
        full_model_path = os.path.join(args.checkpoint, "full_model_state.pt")
        metadata_path = os.path.join(args.checkpoint, "model_metadata.json")
        
        if not (os.path.exists(full_model_path) and os.path.exists(metadata_path)):
            raise FileNotFoundError(f"Enhanced checkpoint not found at {full_model_path} or {metadata_path}")
    
        logger.info("Found enhanced checkpoint with full model state!")
        
        # Load metadata and full state
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        if metadata["num_labels"] != len(label_names):
            raise ValueError(f"Label count mismatch: checkpoint has {metadata['num_labels']}, current dataset has {len(label_names)}")

        model = get_peft_model(model, peft_cfg)
        full_state_dict = torch.load(full_model_path, map_location="cpu")
        model.load_state_dict(full_state_dict, strict=False)
        logger.info("Enhanced checkpoint loaded successfully!")

        checkpoint_for_trainer = args.checkpoint
    else:
        # No checkpoint
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

    # Add compute_metrics function for accuracy calculation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    logger.info("Setting up training arguments")
    # Check if we're on MPS (Apple Silicon)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    training_args = TrainingArguments(
        output_dir          = args.output_dir,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        # Tell Trainer which key in each batch holds the ground‑truth labels.
        # Without it (especially with PEFT/LoRA wrappers), Trainer thinks there
        # are no labels, skips compute_metrics, and never logs eval_accuracy.
        label_names=["labels"],
        eval_strategy      = "steps",
        eval_steps         = 50,
        save_strategy      = "steps",
        save_steps         = 100,
        num_train_epochs   = args.epochs,
        learning_rate      = args.learning_rate,
        weight_decay       = 0.05,
        fp16               = device.type == "cuda",
        save_total_limit   = 3,
        logging_dir        = os.path.join(args.output_dir, "logs"),
        logging_steps      = 10,
        report_to          = "tensorboard",
        load_best_model_at_end = True,
        metric_for_best_model = "eval_accuracy",
        greater_is_better = True,
        # Pass the actual checkpoint path for proper resumption
        resume_from_checkpoint = checkpoint_for_trainer,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = test_dataset,
        data_collator   = collate,
        compute_metrics = compute_metrics,
    )

    # If loading from checkpoint, evaluate immediately to verify the model state
    if args.checkpoint:
        logger.info("Evaluating model immediately after checkpoint loading to verify state...")
        
        # Check which adapter is active (for debugging)
        if hasattr(model, 'active_adapters'):
            logger.info(f"Active adapters: {model.active_adapters}")
        if hasattr(model, 'peft_config'):
            logger.info(f"PEFT config keys: {list(model.peft_config.keys())}")
            
        initial_eval = trainer.evaluate(test_dataset)
        logger.info(f"Initial evaluation after checkpoint loading: {initial_eval}")
    else:
        logger.info("No checkpoint provided, starting fresh training")

    ######################################################################
    # 5. Train & push to the Hub
    # -------------------------------------------------------------------
    logger.info("Starting training")
    if args.checkpoint:
        logger.info(f"Resuming training from checkpoint: {args.checkpoint}")
    trainer.train()
    logger.info("Training complete")
    
    # Evaluate on test set after training
    logger.info("Evaluating on test set")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")

    # Save the final model
    logger.info("Saving final model")
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    
    # Also save the full model state including classification head for PEFT models
    if hasattr(model, 'peft_config'):
        logger.info("Saving full model state for PEFT model...")
        full_model_path = os.path.join(final_model_path, "full_model_state.pt")
        
        # Save the complete model state dict including classification head
        full_state_dict = {}
        for name, param in model.named_parameters():
            full_state_dict[name] = param.data.clone()
        
        torch.save(full_state_dict, full_model_path)
        logger.info(f"Full model state saved to {full_model_path}")
        
        # Save important model metadata
        model_metadata = {
            "num_labels": len(label_names),
            "label_names": label_names,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "base_model": "facebook/dinov2-base-imagenet1k-1-layer"
        }
        
        metadata_path = os.path.join(final_model_path, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=2)
        logger.info(f"Model metadata saved to {metadata_path}")
    
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save the label names for future reference
    label_mapping = {i: label for i, label in enumerate(label_names)}
    with open(os.path.join(final_model_path, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=2)
    logger.info("Label mapping saved")
    
    # trainer.push_to_hub("your‑username/dinov2-font‑classifier")
