import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from font_weight_labels import WEIGHT_GROUP_NAMES, weight_group


def validate_split(root: Path) -> dict:
    family_counts = Counter()
    group_counts = Counter()
    pairs = defaultdict(list)
    missing = []

    metadata_files = sorted(root.glob("*/metadata.jsonl"))
    if not metadata_files:
        raise ValueError(f"No metadata below {root}")

    for metadata in metadata_files:
        for line_number, line in enumerate(metadata.read_text().splitlines(), start=1):
            record = json.loads(line)
            location = f"{metadata}:{line_number}"
            if record["family"] != metadata.parent.name:
                raise ValueError(f"{location}: family does not match its directory")
            expected_group = weight_group(record["weight"])
            if record["weight_group"] != expected_group:
                raise ValueError(f"{location}: incorrect weight group")
            image = metadata.parent / record["file_name"]
            if not image.is_file():
                missing.append(str(image))
            family_counts[record["family"]] += 1
            group_counts[expected_group] += 1
            pairs[record["pair_id"]].append(record)

    if missing:
        raise ValueError(f"Missing {len(missing)} images, first: {missing[0]}")
    for pair_id, records in pairs.items():
        if len(records) > 2:
            raise ValueError(f"{pair_id}: expected at most two records")
        if len({record["family"] for record in records}) != 1:
            raise ValueError(f"{pair_id}: contains multiple families")
        if len(records) == 2 and len({record["weight_group"] for record in records}) != 2:
            raise ValueError(f"{pair_id}: paired records use the same weight group")

    return {
        "families": len(family_counts),
        "images": sum(family_counts.values()),
        "pairs": len(pairs),
        "weight_groups": {
            WEIGHT_GROUP_NAMES[group]: group_counts[group]
            for group in range(len(WEIGHT_GROUP_NAMES))
        },
        "min_images_per_family": min(family_counts.values()),
        "max_images_per_family": max(family_counts.values()),
    }


def pair_ids(root: Path) -> set[str]:
    return {
        json.loads(line)["pair_id"]
        for metadata in root.glob("*/metadata.jsonl")
        for line in metadata.read_text().splitlines()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate paired font dataset metadata")
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    report = {
        split: validate_split(args.data_dir / split)
        for split in ("train", "validation", "test")
    }
    validation_pairs = pair_ids(args.data_dir / "validation")
    test_pairs = pair_ids(args.data_dir / "test")
    overlap = validation_pairs & test_pairs
    if overlap:
        raise ValueError(f"Validation and test share {len(overlap)} pairs")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
