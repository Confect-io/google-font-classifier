import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from safetensors import safe_open
from transformers import (
    AutoImageProcessor,
    Dinov2ForImageClassification,
    Trainer,
    TrainingArguments,
)

from handler import get_inference_transform

logger = logging.getLogger(__name__)

MODEL = "facebook/dinov2-base-imagenet1k-1-layer"

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DINOv2 model for font classification')
    parser.add_argument('--data_dir', type=str, default=None,
                      help='Directory containing the font dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save the model')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training and evaluation batch size')
    parser.add_argument('--epochs', type=int, default=1,
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
    parser.add_argument('--huggingface_model_name', type=str, default=None,
                      help='Name of the model to push to the Hub')
    return parser.parse_args()


def load_checkpoint_with_size_mismatch_handling(base_model, checkpoint_path, peft_config):
    """
    Load PEFT checkpoint with automatic handling of size mismatches.
    This uses PEFT's built-in loading but with strict=False to handle size mismatches gracefully.

    Basically, if we have a different number of labels than in the checkpoint, we re-initialize the classifier head to relearn it.
    
    Args:
        base_model: The base model with the new classifier size
        checkpoint_path: Path to the checkpoint
        peft_config: LoraConfig object with the desired configuration
    
    Returns:
        PeftModel with loaded weights (mismatched layers will be skipped)
    """
    logger.info(f"Loading checkpoint with automatic size mismatch handling: {checkpoint_path}")
    
    try:
        # Try the normal PEFT loading first
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            is_trainable=True
        )
        logger.info("Successfully loaded checkpoint without size mismatches")
        return model
    except Exception as e:
        logger.info(f"Standard loading failed ({str(e)}), using fallback loading method")
        
        # Fallback: Create fresh PEFT model and load compatible weights
        # Note: PeftModel.from_pretrained might have partially modified base_model before failing,
        # so we recreate a clean base model to avoid double-loading warnings
        fresh_base = Dinov2ForImageClassification.from_pretrained(
            MODEL,
            num_labels=base_model.config.num_labels,
            ignore_mismatched_sizes=True,
        )
        
        model = get_peft_model(fresh_base, peft_config)
        
        # Load checkpoint state dict
        checkpoint_file = os.path.join(checkpoint_path, "adapter_model.safetensors")

        if not os.path.exists(checkpoint_file):
            raise ValueError(f"Checkpoint file {checkpoint_file} does not exist")
        
        checkpoint_state_dict = {}
        with safe_open(checkpoint_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint_state_dict[key] = f.get_tensor(key)
    
        # Load only compatible weights
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state_dict, strict=False)
        
        logger.info(f"Loaded checkpoint with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
        logger.info(f"The following keys were in the checkpoint but are now missing: {missing_keys}")
        logger.info(f"The following keys are new i.e. unexpected: {unexpected_keys}")
        logger.info("Missing keys (likely new classifier parameters): will be randomly initialized")
        
        return model


def get_transform(processor: AutoImageProcessor, size: int):
    aug = get_inference_transform(processor, size)

    def transform(example, train=True):
        # Apply the same pad-to-square + resize + normalize pipeline used at inference
        # (defined in handler.py) to ensure no train/serve skew.
        example["pixel_values"] = aug(example["image"])
        return example

    return transform


if __name__ == "__main__":
    args = parse_args()
    
    # Configure logging with timestamps
    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    ######################################################################
    # Directory layout expected by ImageFolder:
    #   fonts/
    #     ├─ Arial/
    #     │   ├─ img001.png
    #     │   └─ ...
    #     ├─ TimesNewRoman/
    #     └─ ...

    logger.info(f"Loading dataset from {args.data_dir}")
    # Get label names from directory names and sort them alphabetically
    # to match the order used by the imagefolder dataset loader
    label_names = sorted(os.listdir(f"{args.data_dir}/train"))
    logger.info(f"Found {len(label_names)} labels")

    if len(label_names) <= 1:
        raise ValueError(f"Expected at least 2 labels, got {label_names=}, imagefolder will not label the dataset if there are less than 2 labels.")

    # READ: the label ids assigned are in alphabetical order.
    train_dataset = None
    test_dataset = None
    

    logger.info("Setting up image processor and augmentations")
    processor   = AutoImageProcessor.from_pretrained(MODEL)  # 224 px
    size        = processor.size["shortest_edge"]            # 224 by default
    
    if args.epochs > 0:
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.data_dir,
        )
        
        logger.info(f"Train size: {len(dataset['train'])}, Validation size: {len(dataset['test'])}")

        transform = get_transform(processor, size)

        logger.info("Applying data transformations")
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

    logger.info("Loading DINOv2 model")
    
    base = Dinov2ForImageClassification.from_pretrained(
            MODEL,
            num_labels=len(label_names),
            ignore_mismatched_sizes=True,
        )

    logger.info("Configuring LoRA adapters")
    peft_cfg = LoraConfig(
        r             = args.lora_rank,
        lora_alpha    = args.lora_alpha,
        target_modules = ["query", "value"],  # Q & V proj in ViT blocks
        lora_dropout  = args.lora_dropout,
        bias          = "none",
        modules_to_save = ["classifier"],  # IMPORTANT: Save classification head too!
    )

    if args.checkpoint:
        model = load_checkpoint_with_size_mismatch_handling(base, args.checkpoint, peft_cfg)
    else:
        model  = get_peft_model(base, peft_cfg)        # fresh LoRA wrap
    
    model.print_trainable_parameters()

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
        eval_strategy      = "steps" if args.epochs > 0 else "no",
        eval_steps         = 500,
        save_strategy      = "steps" if args.epochs > 0 else "no",
        save_steps         = 500,
        num_train_epochs   = args.epochs,
        learning_rate      = args.learning_rate,
        weight_decay       = 0.05,
        fp16               = device.type == "cuda",
        save_total_limit   = 3,
        logging_dir        = os.path.join(args.output_dir, "logs") if args.output_dir else None,
        logging_steps      = 10,
        dataloader_num_workers = 4,
        report_to          = "tensorboard",
        load_best_model_at_end = True,
        metric_for_best_model = "eval_accuracy",
        greater_is_better = True,
        # Pass the actual checkpoint path for proper resumption
        resume_from_checkpoint = args.checkpoint if args.checkpoint else None,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = test_dataset,
        data_collator   = collate,
        compute_metrics = compute_metrics,
    )

    logger.info("Starting training")
    if args.checkpoint:
        logger.info(f"Resuming training from checkpoint: {args.checkpoint}")
    
    if args.epochs > 0:
        trainer.train()
    logger.info("Training complete")
    
    # Saves the result model to the output directory
    # The reason this is important is if we configure load_best_model_at_end=True,
    # the best model will be saved out of all checkpoints.
    # So, even though the trainer already saves the last model as a checkpoint, that one is not necessarily the best.
    if args.output_dir:
        logger.info("Saving result model to the output directory")
        trainer.save_model(f"{args.output_dir}/result_model")

    if args.huggingface_model_name:
        logger.info(f"Pushing model to the Hub: {args.huggingface_model_name}")

        trainer.hub_model_id = args.huggingface_model_name

        with tempfile.TemporaryDirectory() as tmp:
            # Merge the PEFT weights into the base model so that we upload an independent complete model.
            merged = trainer.model.merge_and_unload()
            id2label = {i: name for i, name in enumerate(label_names)}
            label2id = {name: i for i, name in enumerate(label_names)}

            merged.config.id2label = id2label
            merged.config.label2id = label2id
            merged.config.pipeline_tag = "image-classification"
            merged.save_pretrained(tmp, safe_serialization=True)
            processor.save_pretrained(tmp)

            # bundle handler and code
            shutil.copy("handler.py", tmp)
            Path(tmp, "requirements.txt").write_text("\n".join([
                "torchvision>=0.19",
                "Pillow>=10",
            ]))

            HfApi().upload_folder(
                repo_id=args.huggingface_model_name, 
                folder_path=tmp,
                commit_message="Add merged model + processor",
                token=os.environ["HUGGINGFACE_API_KEY"],
            )
