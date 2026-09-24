import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def validation_pair_ids(records: list[dict], fraction: float, seed: int) -> set[str]:
    pairs = defaultdict(list)
    for record in records:
        pairs[record["pair_id"]].append(record)

    singles = sorted(pair_id for pair_id, rows in pairs.items() if len(rows) == 1)
    doubles = sorted(pair_id for pair_id, rows in pairs.items() if len(rows) == 2)
    random.Random(seed).shuffle(singles)
    random.Random(seed + 1).shuffle(doubles)
    target = round(len(records) * fraction)

    choices = []
    for double_count in range(len(doubles) + 1):
        single_count = min(max(target - 2 * double_count, 0), len(singles))
        size = 2 * double_count + single_count
        choices.append((abs(target - size), double_count, single_count))
    _, double_count, single_count = min(choices)
    return set(doubles[:double_count] + singles[:single_count])


def split_evaluation(data_dir: Path, fraction: float, seed: int) -> None:
    test_root = data_dir / "test"
    validation_root = data_dir / "validation"
    if validation_root.exists():
        raise ValueError(f"{validation_root} already exists")

    for metadata in sorted(test_root.glob("*/metadata.jsonl")):
        family = metadata.parent.name
        records = [json.loads(line) for line in metadata.read_text().splitlines()]
        selected = validation_pair_ids(records, fraction, seed)
        validation_records = [row for row in records if row["pair_id"] in selected]
        test_records = [row for row in records if row["pair_id"] not in selected]
        destination = validation_root / family
        destination.mkdir(parents=True)
        for record in validation_records:
            (metadata.parent / record["file_name"]).replace(
                destination / record["file_name"]
            )
        destination.joinpath("metadata.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in validation_records)
        )
        metadata.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in test_records)
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split generated evaluation pairs into validation and locked test sets"
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--validation_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation_fraction must be between zero and one")
    split_evaluation(args.data_dir, args.validation_fraction, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
