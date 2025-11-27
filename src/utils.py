import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed=42):
    """ Ensures reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#set_seed()

def pairwise_distance(embeddings):
    """Compute (N x N) pairwise Euclidean distance matrix"""
    dot = torch.matmul(embeddings, embeddings.t())
    sq = torch.sum(embeddings ** 2, dim=1, keepdim=True)
    dist = sq + sq.t() - 2 * dot
    dist = torch.clamp(dist, min=1e-12)
    return torch.sqrt(dist)

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.rankloss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, labels):
        N = embeddings.size(0)
        dist = pairwise_distance(embeddings)

        labels = labels.unsqueeze(1)
        is_pos = labels.eq(labels.t())
        is_neg = labels.ne(labels.t())

        # hardest positive
        max_pos = (dist * is_pos.float()).max(1)[0]

        # hardest negative
        dist_neg = dist + (is_pos.float() * 1e6)
        min_neg = dist_neg.min(1)[0]

        y = torch.ones_like(min_neg)
        return self.rankloss(min_neg, max_pos, y)

