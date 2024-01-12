import torch
import torch.nn.functional as F

import numpy as np

class ModernHopfieldNetwork:
    def __init__(self, K):
        self.K = K

    def __call__(self, query, beta, use_cosine_similarity=True, sample=False, device=None):

        assert len(query.shape) == 2 # query must be of shape (batch, dim)

        # Compute similarity scores
        if use_cosine_similarity:
            keys = self.K / torch.sum(self.K, dim=1).unsqueeze(-1)
        else:
            keys = self.K
        sim_score = torch.matmul(query, keys.T)

        # Separate similarity scores
        sim_score = F.softmax(beta * sim_score, dim=1)

        if sample:
            choices = torch.multinomial(sim_score, 1)
            sim_score = F.one_hot(choices, sim_score.size(1)).float()


        out = torch.matmul(sim_score, self.K)
        return out