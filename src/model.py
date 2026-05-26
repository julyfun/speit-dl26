"""Fusion model: transformer encoder + metadata MLP."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class FusionClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_metadata: int,
        use_metadata: bool = True,
        dropout: float = 0.15,
        meta_hidden: int = 64,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.use_metadata = use_metadata
        config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        # 分类用 last_hidden_state[:, 0]，不用 pooler；保留 pooler 会在 DDP 下报 unused param
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=config,
            add_pooling_layer=False,
            local_files_only=local_files_only,
        )
        hidden_size = config.hidden_size

        if use_metadata:
            self.meta_mlp = nn.Sequential(
                nn.Linear(num_metadata, meta_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(meta_hidden, meta_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            classifier_in = hidden_size + meta_hidden
        else:
            self.meta_mlp = None
            classifier_in = hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_in, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0]

        if self.use_metadata:
            if metadata is None:
                raise ValueError("metadata is required when use_metadata=True")
            meta_hidden = self.meta_mlp(metadata)
            fused = torch.cat([cls_hidden, meta_hidden], dim=-1)
        else:
            fused = cls_hidden

        fused = self.dropout(fused)
        return self.classifier(fused)
