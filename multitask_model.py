from dataclasses import dataclass

import torch
import torch.nn.functional as F
from font_weight_labels import WEIGHT_THRESHOLDS
from torch import nn
from transformers import Dinov2ForImageClassification
from transformers.utils import ModelOutput


@dataclass
class FontClassifierOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    weight_logits: torch.Tensor | None = None
    family_loss: torch.Tensor | None = None
    weight_loss: torch.Tensor | None = None
    consistency_loss: torch.Tensor | None = None


class Dinov2ForFontClassification(Dinov2ForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        thresholds = tuple(
            getattr(config, "font_weight_thresholds", WEIGHT_THRESHOLDS)
        )
        config.font_weight_thresholds = list(thresholds)
        config.keys_to_ignore_at_inference = [
            "family_loss",
            "weight_loss",
            "consistency_loss",
        ]
        self.ordinal_head = nn.Linear(
            self.classifier.in_features, len(thresholds)
        )
        nn.init.trunc_normal_(self.ordinal_head.weight, std=0.02)
        nn.init.zeros_(self.ordinal_head.bias)

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        sequence = self.dinov2(pixel_values=pixel_values).last_hidden_state
        return torch.cat((sequence[:, 0], sequence[:, 1:].mean(dim=1)), dim=1)

    @staticmethod
    def _family_consistency(
        family_logits: torch.Tensor, pair_ids: torch.Tensor
    ) -> torch.Tensor | None:
        losses = []
        for pair_id in pair_ids.unique():
            if pair_id < 0:
                continue
            selected = family_logits[pair_ids == pair_id]
            if selected.shape[0] < 2:
                continue
            log_probabilities = F.log_softmax(selected, dim=-1)
            probabilities = log_probabilities.exp()
            mean = probabilities.mean(dim=0).clamp_min(1e-8)
            losses.append(
                (probabilities * (log_probabilities - mean.log())).sum(dim=-1).mean()
            )
        return torch.stack(losses).mean() if losses else None

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        weight_labels: torch.Tensor | None = None,
        pair_ids: torch.Tensor | None = None,
        weight_loss_scale: float = 1.0,
        consistency_loss_scale: float = 1.0,
        **_: object,
    ) -> FontClassifierOutput:
        features = self._features(pixel_values)
        family_logits = self.classifier(features)
        weight_logits = self.ordinal_head(features)

        family_loss = None
        if labels is not None and torch.any(labels >= 0):
            family_loss = F.cross_entropy(family_logits, labels, ignore_index=-100)

        weight_loss = None
        if weight_labels is not None:
            valid = weight_labels >= 0
            if torch.any(valid):
                ranks = torch.arange(
                    1, weight_logits.shape[1] + 1, device=weight_logits.device
                )
                targets = (weight_labels[valid, None] >= ranks).to(weight_logits.dtype)
                weight_loss = F.binary_cross_entropy_with_logits(
                    weight_logits[valid], targets
                )

        consistency_loss = None
        if pair_ids is not None:
            consistency_loss = self._family_consistency(family_logits, pair_ids)

        losses = [loss for loss in (family_loss,) if loss is not None]
        if weight_loss is not None:
            losses.append(weight_loss_scale * weight_loss)
        if consistency_loss is not None:
            losses.append(consistency_loss_scale * consistency_loss)
        loss = torch.stack(losses).sum() if losses else None

        return FontClassifierOutput(
            loss=loss,
            logits=family_logits,
            weight_logits=weight_logits,
            family_loss=family_loss,
            weight_loss=weight_loss,
            consistency_loss=consistency_loss,
        )


class FontOnnxWrapper(nn.Module):
    def __init__(self, model: Dinov2ForFontClassification):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        output = self.model(pixel_values=pixel_values)
        return output.logits, output.weight_logits
