import json
from pathlib import Path

import numpy as np
import pytest
import torch
from font_weight_labels import ordinal_targets, weight_group
from multitask_dataset import PairedFontDataset, family_names
from multitask_model import Dinov2ForFontClassification
from PIL import Image
from split_multitask_evaluation import validation_pair_ids
from train_multitask import metrics


@pytest.mark.parametrize(
    ("weight", "group"),
    [(300, 0), (400, 0), (500, 1), (600, 1), (700, 2), (900, 2)],
)
def test_weight_groups(weight, group):
    assert weight_group(weight) == group


def test_ordinal_targets():
    assert ordinal_targets(0) == (0.0, 0.0)
    assert ordinal_targets(1) == (1.0, 0.0)
    assert ordinal_targets(2) == (1.0, 1.0)


def test_weights_outside_training_scope_are_rejected():
    with pytest.raises(ValueError):
        weight_group(200)


def _write_pair(root, split, family, pair_id, weights):
    directory = root / split / family
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, weight in enumerate(weights):
        filename = f"{index}-w{weight}.jpg"
        Image.new("RGB", (8, 8), "white").save(directory / filename)
        records.append(
            {
                "file_name": filename,
                "family": family,
                "weight": weight,
                "weight_group": weight_group(weight),
                "pair_id": pair_id,
            }
        )
    (directory / "metadata.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def test_dataset_preserves_pairs_and_masks_external_families(tmp_path):
    known = tmp_path / "known"
    external = tmp_path / "external"
    for split in ("train", "test"):
        _write_pair(known, split, "Known", f"known-{split}", [400, 700])
        _write_pair(external, split, "External", f"external-{split}", [300, 600])

    assert family_names(known) == ["Known"]
    dataset = PairedFontDataset(known, "train", ["Known"], external)
    pairs = [dataset[index] for index in range(len(dataset))]
    assert sorted(len(pair) for pair in pairs) == [2, 2]
    assert sorted({item["family_label"] for pair in pairs for item in pair}) == [-100, 0]


def test_evaluation_split_keeps_pairs_together():
    records = [
        {"pair_id": "a"},
        {"pair_id": "a"},
        {"pair_id": "b"},
        {"pair_id": "b"},
        {"pair_id": "c"},
        {"pair_id": "d"},
    ]

    selected = validation_pair_ids(records, 0.5, 42)

    validation = [row for row in records if row["pair_id"] in selected]
    test = [row for row in records if row["pair_id"] not in selected]
    assert len(validation) == 3
    assert {row["pair_id"] for row in validation}.isdisjoint(
        row["pair_id"] for row in test
    )


def test_family_consistency_is_zero_for_equal_distributions():
    logits = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    pair_ids = torch.tensor([0, 0, 1])
    loss = Dinov2ForFontClassification._family_consistency(logits, pair_ids)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_family_consistency_penalizes_weight_dependent_predictions():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    pair_ids = torch.tensor([0, 0])
    loss = Dinov2ForFontClassification._family_consistency(logits, pair_ids)
    assert loss.item() > 0.5


def test_metrics_report_both_heads_joint_result_and_pair_stability():
    family_logits = np.array([[5, 0], [5, 0], [0, 5], [0, 5]])
    weight_logits = np.array([[-5, -5], [5, 5], [-5, -5], [5, 5]])
    labels = np.array([0, 0, 1, 1])
    weight_labels = np.array([0, 2, 0, 2])
    pair_ids = np.array([0, 0, 1, 1])

    result = metrics(
        ((family_logits, weight_logits), (labels, weight_labels, pair_ids))
    )

    assert result["family_accuracy"] == 1.0
    assert result["weight_group_accuracy"] == 1.0
    assert result["joint_accuracy"] == 1.0
    assert result["paired_family_stability"] == 1.0
    assert result["family_accuracy_700_800_900"] == 1.0


def test_export_reconstructs_the_initial_adapter(monkeypatch):
    import export_multitask_onnx as exporter

    calls = []

    class FakeModel:
        def __init__(self, name):
            self.name = name

        def merge_and_unload(self):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        exporter.Dinov2ForFontClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel("base"),
    )

    def from_pretrained(base, adapter, **kwargs):
        calls.append((base.name, adapter, kwargs))
        return FakeModel(f"{base.name}+{adapter}")

    monkeypatch.setattr(exporter.PeftModel, "from_pretrained", from_pretrained)

    result = exporter.load_model(
        Path("weight-adapter"),
        {
            "family_labels": ["A", "B"],
            "initial_adapter": "family-adapter",
            "initial_adapter_subfolder": "result_model",
        },
    )

    assert calls == [
        ("base", "family-adapter", {"subfolder": "result_model"}),
        ("base+family-adapter", Path("weight-adapter"), {}),
    ]
    assert result.name == "base+family-adapter+weight-adapter"
