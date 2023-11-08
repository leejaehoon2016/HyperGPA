from torch import nn
from torch.nn.modules.linear import Linear

"""
must
first : in_feats 
second : out_feats
"""
class L(nn.Module):
    def __init__(self,in_feats, out_feats, **karg):
        super().__init__()
        self.layer = Linear(in_feats,out_feats)
    def forward(self,x):
        return self.layer(x)

class LT(nn.Module):
    def __init__(self,in_feats, out_feats, **karg):
        super().__init__()
        self.layer = nn.Sequential(Linear(in_feats,out_feats), nn.Tanh())
    def forward(self,x):
        return self.layer(x)

class LRL(nn.Module):
    def __init__(self,in_feats, out_feats, mid_dims, **karg):
        super().__init__()
        layer = []
        in_dim = in_feats
        for next_dim in mid_dims:
            layer.append(nn.Linear(in_dim, next_dim))
            layer.append(nn.ReLU())
            in_dim = next_dim
        layer.append(nn.Linear(in_dim, out_feats))
        self.layers = nn.Sequential(*layer)
    def forward(self,x):
        return self.layers(x)

class LTLL(nn.Module):
    def __init__(self,in_feats, out_feats, mid_dim, **karg):
        super().__init__()
        layer = []
        layer.append(Linear(in_feats,mid_dim))
        layer.append(nn.Tanh())
        layer.append(Linear(mid_dim,mid_dim))
        layer.append(Linear(mid_dim,out_feats))
        self.layers = nn.Sequential(*layer)
    def forward(self,x):
        return self.layers(x)


class LTL(nn.Module):
    def __init__(self,in_feats, out_feats, mid_dims, **karg):
        super().__init__()
        layer = []
        in_dim = in_feats
        for next_dim in mid_dims:
            layer.append(nn.Linear(in_dim, next_dim))
            layer.append(nn.Tanh())
            in_dim = next_dim
        layer.append(nn.Linear(in_dim, out_feats))
        self.layers = nn.Sequential(*layer)
    def forward(self,x):
        return self.layers(x)
