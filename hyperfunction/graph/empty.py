import torch
import dgl
from torch import nn

"""
first arg have to be in_feats
output_dim attribute is needed
"""
class Empty(torch.nn.Module):
    def __init__(self, in_feats, **karg):
        super().__init__()
        self.output_dim = in_feats
        
    def forward(self, graph, node_feature):
        return node_feature
