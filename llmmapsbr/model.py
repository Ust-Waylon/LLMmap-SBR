"""LLMmap: MLP mapping with a last-skip self-attention session encoder."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-5)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width, eps=1e-5)
        self.ff = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask, occupied):
        h = self.norm1(x)
        attention, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = (x + self.drop(attention)) * occupied
        return (x + self.drop(self.ff(self.norm2(x)))) * occupied


class SessionEncoder(nn.Module):
    def __init__(self, width=256, layers=2, heads=4, dropout=0.2, max_length=20):
        super().__init__()
        self.max_length = max_length
        self.pos = nn.Embedding(max_length, width)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([AttentionBlock(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width, eps=1e-5)

    def forward(self, embeddings, valid):
        length = embeddings.shape[1]
        if length > self.max_length:
            raise ValueError("History exceeds the configured attention window.")
        occupied = valid.unsqueeze(-1).to(embeddings.dtype)
        positions = torch.arange(length, device=embeddings.device)
        x = self.drop((embeddings + self.pos(positions)) * occupied)
        # Padded states are zeroed after each residual addition, as in the paper.
        causal = torch.zeros(length, length, device=x.device, dtype=x.dtype)
        future = torch.triu(torch.ones_like(causal, dtype=torch.bool), diagonal=1)
        causal = causal.masked_fill(future, float("-inf"))
        for layer in self.layers:
            x = layer(x, causal, occupied)
        return self.norm(x) * occupied


class LLMmap(nn.Module):
    def __init__(self, input_dim=2560, output_dim=256, hidden_dim=1280,
                 max_length=20, layers=2, heads=4, dropout=0.2, gamma_init=0.05, temperature=0.07):
        super().__init__()
        if temperature <= 0 or max_length < 1 or output_dim < 1:
            raise ValueError("Temperature, history length, and embedding width must be positive.")
        self.temperature = float(temperature)
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
        self.encoder = SessionEncoder(output_dim, layers, heads, dropout, max_length)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def item_table(self, language_vectors):
        return F.normalize(self.mapping(language_vectors), dim=-1)

    def session_embedding(self, mapped_items, history):
        valid = history >= 0
        if not bool(valid[:, -1].all()):
            raise ValueError("Each left-padded history must end in a valid item.")
        embeddings = mapped_items[history.clamp(min=0)] * valid.unsqueeze(-1)
        last = embeddings[:, -1]
        attention = self.encoder(embeddings, valid)[:, -1]
        return F.normalize(last + self.gamma * F.normalize(attention, dim=-1), dim=-1)

    def forward(self, language_vectors, history, *, mapped_items=None):
        mapped_items = self.item_table(language_vectors) if mapped_items is None else mapped_items
        session = self.session_embedding(mapped_items, history)
        scores = session @ mapped_items.T / self.temperature
        valid = history >= 0
        row = torch.arange(history.size(0), device=history.device)[:, None].expand_as(history)
        scores[row[valid], history[valid]] = -1e9
        return scores
