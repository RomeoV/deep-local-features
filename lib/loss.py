
import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class SimilarityLoss(nn.Module):

    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, attentions1, attentions2, correspondence):
        return F.mse_loss(attention1, attention2)