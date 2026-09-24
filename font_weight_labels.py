from pathlib import Path

WEIGHT_THRESHOLDS = (500, 700)
WEIGHT_GROUP_NAMES = (
    "300/400",
    "500/600",
    "700/800/900",
)


def font_weight(path: str | Path) -> int:
    from fontTools.ttLib import TTFont

    font = TTFont(str(path), lazy=True)
    try:
        weight = int(font["OS/2"].usWeightClass)
    finally:
        font.close()
    if weight < 1 or weight > 1000:
        raise ValueError(f"{path}: invalid OS/2 weight {weight}")
    return weight


def weight_group(weight: int) -> int:
    if weight < 300 or weight > 900:
        raise ValueError(f"weight {weight} is outside the 300-900 training scope")
    return sum(weight >= threshold for threshold in WEIGHT_THRESHOLDS)


def ordinal_targets(group: int) -> tuple[float, ...]:
    if group < 0 or group >= len(WEIGHT_GROUP_NAMES):
        raise ValueError(f"invalid weight group {group}")
    return tuple(float(group >= rank) for rank in range(1, len(WEIGHT_GROUP_NAMES)))
