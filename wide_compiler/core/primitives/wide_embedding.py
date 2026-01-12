from typing import List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

class WideEmbedding(nn.Module):
    """N parallel Embedding tables."""

    def __init__(self, n: int, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.n = n
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Single embedding, output is N*D per token
        # Each model has its own D-dim embedding
        self.weight = nn.Parameter(torch.randn(n, num_embeddings, embedding_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T] indices (same for all N models)
        # out: [B, T, N*D]
        B, T = x.shape

        # Gather from each table: [N, B, T, D]
        out = nn.embedding(x, self.weight[0])  # [B, T, D]
        outs = [F.embedding(x, self.weight[i]) for i in range(self.n)]

        # Stack and reshape: [B, T, N*D]
        out = torch.cat(outs, dim=-1)
        return out

    @classmethod
    def from_modules(cls, modules: List[nn.Embedding]) -> 'WideEmbedding':
        n = len(modules)
        t = modules[0]

        wide = cls(n, t.num_embeddings, t.embedding_dim)

        with torch.no_grad():
            for i, m in enumerate(modules):
                wide.weight[i] = m.weight

        return wide

    def __repr__(self):
        return f"WideEmbedding({self.n}x[{self.num_embeddings}, {self.embedding_dim}])"
