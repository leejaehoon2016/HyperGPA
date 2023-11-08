import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
"""
first arg have to be in_feats
output_dim attribute is needed
"""
class AVWGCN(nn.Module):
    def __init__(self, in_feats, out_feats, node_num, cheb_k, embed_dim, **kargs):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.output_dim = out_feats
        self.weights_pool = nn.Parameter(torch.randn(embed_dim, cheb_k, in_feats, out_feats))
        self.bias_pool = nn.Parameter(torch.randn(embed_dim, out_feats))
        self.node_embeddings = nn.Parameter(torch.randn(node_num, embed_dim), requires_grad=True)
    def forward(self, graph, node_feature):
        # if node_feature.isnan().sum() == 0:
        #     print(node_feature)
        # import pdb ; pdb.set_trace()
        # x = node_feature.unsqueeze(0)
        x = node_feature
        node_embeddings = self.node_embeddings
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        # import pdb ; pdb.set_trace()
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        # if node_feature.isnan().sum() == 0:
        # if x_gconv.isnan().sum() == 0:
        #     print("pass")
        # else:
        #     print("not pass")
            # print(x_gconv)
            # pass
        # import pdb ; pdb.set_trace()
        return x_gconv