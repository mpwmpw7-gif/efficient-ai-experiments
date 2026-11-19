"""Frozen MiniLM role features and the original five-class PyTorch readout."""
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


def mentions(text, phrase):
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return [match.span() for match in re.finditer(pattern, text, re.I)]


def intersects(offsets, spans):
    return np.array([end > start and any(start < high and end > low for low, high in spans)
                     for start, end in offsets])


class Features:
    def __init__(self, model, device, revision=None):
        self.device = device
        opts = {"local_files_only": True}
        if revision is not None:
            opts["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model, **opts)
        self.encoder = AutoModel.from_pretrained(model, **opts).eval().requires_grad_(False).to(device)

    @torch.no_grad()
    def batch(self, docs):
        tokens = self.tokenizer([row["document"] for row in docs], padding=True,
                                return_offsets_mapping=True, return_tensors="pt")
        offsets = tokens.pop("offset_mapping").numpy()
        if tokens.input_ids.shape[1] > self.encoder.config.max_position_embeddings:
            raise ValueError("A fixture exceeds the encoder's token budget.")
        hidden = self.encoder(**tokens.to(self.device)).last_hidden_state
        masks = []
        for row, off in zip(docs, offsets):
            text = row["document"]
            spans = [(span["start"], span["end"]) for span in row["spans"]]
            for action in row["actions"]:
                act = mentions(text, action)
                for name in row["candidates"]:
                    ref = mentions(text, name)
                    blocks = []
                    for low, high in spans:
                        aa = [(a, b) for a, b in act if low <= a < high]
                        rr = [(a, b) for a, b in ref if low <= a < high]
                        if aa or rr:
                            blocks.append((low, high, aa, rr))
                    shared = [block for block in blocks if block[2] and block[3]]
                    ordered = [[], []]
                    for low, high, aa, rr in shared:
                        for start_a, _ in aa:
                            for start_r, _ in rr:
                                ordered[int(start_a > start_r)].append((low, high))
                    context = [(low, high) for low, high, _, _ in (shared or blocks)]
                    mask = np.stack([intersects(off, part) for part in (act, ref, context, *ordered)])
                    if not mask[:3].any(axis=1).all():
                        raise ValueError(f"A role or context has no matching tokens for {name!r}.")
                    masks.append(mask)
        m = torch.as_tensor(np.stack(masks), device=self.device, dtype=hidden.dtype)
        m = m.reshape(len(docs), 30, -1)
        pooled = (m / m.sum(dim=-1, keepdim=True).clamp_min(1)) @ hidden
        pooled = F.normalize(pooled.reshape(len(docs), 6, 5, 384), dim=-1)
        return torch.cat([pooled.flatten(2), (pooled[:, :, 0] * pooled[:, :, 1]) * 16], dim=-1)

    def encode(self, rows, batch_size):
        fields = ("document", "actions", "candidates", "spans")
        docs = [{k: row[k] for k in fields} for row in rows]
        return torch.cat([self.batch(docs[i:i+batch_size]) for i in range(0, len(docs), batch_size)])


class RelationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(2304, 96), nn.GELU(), nn.Linear(96, 32))
        self.edge = nn.Linear(32, 5)

    def forward(self, features):
        embedding = F.normalize(self.project(features), dim=-1)
        return self.edge(8 * embedding), embedding
