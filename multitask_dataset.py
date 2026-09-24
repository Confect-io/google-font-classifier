import json
from collections import defaultdict
from pathlib import Path

import torch
from handler import get_inference_transform
from PIL import Image
from torch.utils.data import Dataset


def family_names(data_dir: str | Path) -> list[str]:
    root = Path(data_dir) / "train"
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "metadata.jsonl").exists()
    )


def _records(root: Path) -> list[dict]:
    records = []
    for metadata in sorted(root.glob("*/metadata.jsonl")):
        for line in metadata.read_text().splitlines():
            record = json.loads(line)
            record["path"] = metadata.parent / record["file_name"]
            records.append(record)
    if not records:
        raise ValueError(f"No metadata.jsonl records below {root}")
    return records


class PairedFontDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        labels: list[str],
        extra_weight_dir: str | Path | None = None,
    ):
        label_ids = {name: index for index, name in enumerate(labels)}
        records = _records(Path(data_dir) / split)
        if extra_weight_dir is not None:
            extra_root = Path(extra_weight_dir) / split
            if extra_root.exists():
                records.extend(_records(extra_root))

        pairs = defaultdict(list)
        for record in records:
            record["family_label"] = label_ids.get(record["family"], -100)
            pairs[record["pair_id"]].append(record)
        self.pairs = [pairs[key] for key in sorted(pairs)]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> list[dict]:
        result = []
        for record in self.pairs[index]:
            with Image.open(record["path"]) as image:
                item = dict(record)
                item["image"] = image.convert("RGB").copy()
                item["pair_index"] = index
            result.append(item)
        return result


def make_collator(processor, size: int, weight_loss_scale: float, consistency_loss_scale: float):
    transform = get_inference_transform(processor, size)

    def collate(batch: list[list[dict]]) -> dict:
        items = [item for pair in batch for item in pair]
        return {
            "pixel_values": torch.stack(
                [transform(item["image"]) for item in items]
            ),
            "labels": torch.tensor(
                [item["family_label"] for item in items], dtype=torch.long
            ),
            "weight_labels": torch.tensor(
                [item["weight_group"] for item in items], dtype=torch.long
            ),
            "pair_ids": torch.tensor(
                [item["pair_index"] for item in items], dtype=torch.long
            ),
            "weight_loss_scale": weight_loss_scale,
            "consistency_loss_scale": consistency_loss_scale,
        }

    return collate
