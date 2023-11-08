from torch import nn
import torch 
import numpy as np
        
def flatten_dim(feats):
    return np.prod(feats)

class LRL(nn.Module):
    def __init__(self, in_feats, out_feats, mid_dims, **karg):
        super().__init__()
        self.in_feats, self.out_feats = in_feats, out_feats
        try:
            self.in_feats_dimlen  = len(in_feats)
        except:
            self.in_feats_dimlen  = 1
        try:
            self.out_feats_dimlen = len(list(out_feats))
        except:
            self.out_feats_dimlen  = 1
        in_feats, out_feats = flatten_dim(in_feats), flatten_dim(out_feats)
        layer = []
        in_dim = in_feats
        for next_dim in mid_dims:
            layer.append(nn.Linear(in_dim, next_dim))
            layer.append(nn.ReLU())
            in_dim = next_dim
        layer.append(nn.Linear(in_dim, out_feats))
        self.layers = nn.Sequential(*layer)
    def forward(self,x):
        if self.in_feats_dimlen > 1:
            x = x.flatten(start_dim=-self.in_feats_dimlen)
        out = self.layers(x)
        if self.out_feats_dimlen > 1:
            out = out.view(*out.shape[:-1], *self.out_feats)
        return out

class CustomBatchnorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = nn.BatchNorm1d(dim)
    def forward(self,x):
        assert len(x.size()) == 3, "Only x=(batch, state, emb) case is Implemented"
        num_batch = x.size(0)
        num_state = x.size(1)
        x = torch.cat([x[i] for i in range(len(x))],dim=0) 
        x = self.model(x)
        last = torch.stack([x[i*num_state:(i+1)*num_state] for i in range(num_batch)],dim=0) # (batch, num_state, ...) 
        return last



class LRL_DB(nn.Module):
    def __init__(self, in_feats, out_feats, mid_dims, drop=0.2, batch=True, **karg):
        super().__init__()
        self.in_feats, self.out_feats = in_feats, out_feats
        try:
            self.in_feats_dimlen  = len(in_feats)
        except:
            self.in_feats_dimlen  = 1
        try:
            self.out_feats_dimlen = len(list(out_feats))
        except:
            self.out_feats_dimlen  = 1
        in_feats, out_feats = flatten_dim(in_feats), flatten_dim(out_feats)
        layer = []
        in_dim = in_feats
        for i, next_dim in enumerate(mid_dims,1):
            layer.append(nn.Linear(in_dim, next_dim))
            if batch:
                layer.append(CustomBatchnorm(next_dim))
            layer.append(nn.ReLU())
            if drop != 0:
                layer.append(nn.Dropout(p=drop))
            in_dim = next_dim
        layer.append(nn.Linear(in_dim, out_feats))
        self.layers = nn.Sequential(*layer)
    def forward(self,x):
        if self.in_feats_dimlen > 1:
            x = x.flatten(start_dim=-self.in_feats_dimlen)
        out = self.layers(x)
        if self.out_feats_dimlen > 1:
            out = out.view(*out.shape[:-1], *self.out_feats)
        return out
