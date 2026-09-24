import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from font_weight_labels import WEIGHT_GROUP_NAMES
from multitask_dataset import PairedFontDataset, family_names, make_collator
from multitask_model import Dinov2ForFontClassification
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoImageProcessor, Trainer, TrainingArguments

BASE_MODEL = "facebook/dinov2-base-imagenet1k-1-layer"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train independent font-family and ordinal-weight heads"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--extra_weight_data_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--initial_adapter")
    parser.add_argument("--initial_adapter_subfolder")
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--weight_loss_scale", type=float, default=1.0)
    parser.add_argument("--consistency_loss_scale", type=float, default=1.0)
    parser.add_argument("--dataloader_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frozen_probe", action="store_true")
    return parser.parse_args()


def metrics(eval_prediction):
    predictions, targets = eval_prediction
    family_logits, weight_logits = predictions[:2]
    family_labels, weight_labels, pair_ids = targets[:3]
    valid_family = family_labels >= 0
    valid_weight = weight_labels >= 0
    valid_both = valid_family & valid_weight
    family_predictions = family_logits.argmax(axis=-1)
    weight_predictions = (weight_logits >= 0).sum(axis=-1)
    top_k = min(5, family_logits.shape[1])
    top_family = np.argpartition(family_logits, -top_k, axis=-1)[:, -top_k:]

    selected_family_logits = family_logits[valid_family]
    selected_family_labels = family_labels[valid_family]
    maxima = selected_family_logits.max(axis=-1)
    family_cross_entropy = np.mean(
        maxima
        + np.log(np.exp(selected_family_logits - maxima[:, None]).sum(axis=-1))
        - selected_family_logits[
            np.arange(selected_family_labels.shape[0]), selected_family_labels
        ]
    )

    selected_weight_logits = weight_logits[valid_weight]
    ranks = np.arange(1, weight_logits.shape[1] + 1)
    ordinal = (weight_labels[valid_weight, None] >= ranks).astype(np.float32)
    weight_binary_cross_entropy = np.mean(
        np.maximum(selected_weight_logits, 0)
        - selected_weight_logits * ordinal
        + np.log1p(np.exp(-np.abs(selected_weight_logits)))
    )

    consistency_losses = []
    stable_pairs = []
    for pair_id in np.unique(pair_ids):
        selected = family_logits[pair_ids == pair_id]
        predicted = family_predictions[pair_ids == pair_id]
        if selected.shape[0] < 2:
            continue
        shifted = selected - selected.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        mean = np.clip(probabilities.mean(axis=0), 1e-8, None)
        consistency_losses.append(
            np.mean(
                np.sum(
                    probabilities
                    * (np.log(np.clip(probabilities, 1e-8, None)) - np.log(mean)),
                    axis=-1,
                )
            )
        )
        stable_pairs.append(np.all(predicted == predicted[0]))

    result = {
        "family_accuracy": float(
            np.mean(family_predictions[valid_family] == family_labels[valid_family])
        ),
        "family_top5_accuracy": float(
            np.mean(
                np.any(
                    top_family[valid_family]
                    == family_labels[valid_family, None],
                    axis=-1,
                )
            )
        ),
        "weight_group_accuracy": float(
            np.mean(weight_predictions[valid_weight] == weight_labels[valid_weight])
        ),
        "weight_group_distance": float(
            np.mean(
                np.abs(weight_predictions[valid_weight] - weight_labels[valid_weight])
            )
        ),
        "joint_accuracy": float(
            np.mean(
                (family_predictions[valid_both] == family_labels[valid_both])
                & (weight_predictions[valid_both] == weight_labels[valid_both])
            )
        ),
        "family_loss": float(family_cross_entropy),
        "weight_loss": float(weight_binary_cross_entropy),
    }
    if consistency_losses:
        result["consistency_loss"] = float(np.mean(consistency_losses))
        result["paired_family_stability"] = float(np.mean(stable_pairs))
    for group, name in enumerate(WEIGHT_GROUP_NAMES):
        selected = valid_family & (weight_labels == group)
        if np.any(selected):
            metric_name = name.replace("/", "_")
            result[f"family_accuracy_{metric_name}"] = float(
                np.mean(family_predictions[selected] == family_labels[selected])
            )
            result[f"weight_accuracy_{metric_name}"] = float(
                np.mean(weight_predictions[selected] == weight_labels[selected])
            )
    return result


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.labels:
        raw_labels = json.loads(args.labels.read_text())
        labels = (
            [raw_labels[str(index)] for index in range(len(raw_labels))]
            if isinstance(raw_labels, dict)
            else raw_labels
        )
    else:
        labels = family_names(args.data_dir)
    if len(labels) < 2:
        raise ValueError(f"Expected at least two family labels, got {labels}")
    unknown = sorted(set(family_names(args.data_dir)) - set(labels))
    if unknown:
        raise ValueError(f"Dataset families absent from label map: {unknown}")

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    size = processor.size["shortest_edge"]
    datasets = {
        split: PairedFontDataset(
            args.data_dir, split, labels, args.extra_weight_data_dir
        )
        for split in ("train", "test")
    }
    collator = make_collator(
        processor,
        size,
        args.weight_loss_scale,
        args.consistency_loss_scale,
    )

    base = Dinov2ForFontClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    )
    if args.initial_adapter:
        adapter_args = (
            {"subfolder": args.initial_adapter_subfolder}
            if args.initial_adapter_subfolder
            else {}
        )
        base = PeftModel.from_pretrained(
            base, args.initial_adapter, **adapter_args
        ).merge_and_unload()

    if args.frozen_probe:
        if not args.initial_adapter:
            raise ValueError("--frozen_probe requires --initial_adapter")
        for parameter in base.parameters():
            parameter.requires_grad = False
        for parameter in base.ordinal_head.parameters():
            parameter.requires_grad = True
        model = base
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier", "ordinal_head"],
            ),
        )
        model.print_trainable_parameters()

    model.config.id2label = {index: label for index, label in enumerate(labels)}
    model.config.label2id = {label: index for index, label in enumerate(labels)}

    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.05,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_joint_accuracy",
        greater_is_better=True,
        save_total_limit=3,
        logging_steps=10,
        dataloader_num_workers=args.dataloader_workers,
        remove_unused_columns=False,
        label_names=["labels", "weight_labels", "pair_ids"],
        report_to="tensorboard",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=collator,
        compute_metrics=metrics,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    result_dir = output_dir / "result_model"
    trainer.save_model(result_dir)
    processor.save_pretrained(result_dir)
    (result_dir / "font_model_metadata.json").write_text(
        json.dumps(
            {
                "family_labels": labels,
                "weight_thresholds": model.config.font_weight_thresholds,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
