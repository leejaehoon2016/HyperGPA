import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

"""
first arg have to be in_feats
output_dim attribute is needed
input : batch node feature matrix -> (batch, n, input_dim)
output : latent feature matrix -> (batch, n, output_dim)
"""

class GCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats, depth, mid_dim, **karg):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.depth = depth
        in_dim = in_feats
        self.output_dim = in_feats + out_feats * depth
        for i in range(depth-1):
            self.layers.append(dgl.nn.pytorch.conv.GraphConv(in_dim, mid_dim, activation=nn.ReLU()))
            in_dim = mid_dim
        self.layers.append(dgl.nn.pytorch.conv.GraphConv(in_dim, out_feats))
        
        self.main_graph = None
        self.main_batch = None

    def make_graph(self, graph, batch_num):
        num_node = len(graph.nodes())
        src, dst = graph.edges()
        src = src.repeat(batch_num, 1) + torch.Tensor([[num_node*i] for i in range(batch_num)]).to(src.device)
        dst = dst.repeat(batch_num, 1) + torch.Tensor([[num_node*i] for i in range(batch_num)]).to(src.device)
        src = src.flatten()
        dst = dst.flatten()
        graph = dgl.graph((list(src),list(dst)), num_nodes = num_node * batch_num).to(src.device)
        return graph
        
    def forward(self, graph, node_feature):
        batch_num = len(node_feature)
        num_node = len(graph.nodes())
        if self.main_graph is None:
            self.main_graph = self.make_graph(graph,batch_num)
            self.main_batch = batch_num
        
        if self.main_batch != batch_num:
            graph = self.make_graph(graph,batch_num)
        else:
            graph = self.main_graph
        
        node_feature = node_feature.flatten(start_dim=0,end_dim=1)
        graph_feature = self.each_forward(graph, node_feature)
        graph_feature = [graph_feature[i*num_node:(i+1)*num_node] for i in range(batch_num)]
        graph_feature = torch.stack(graph_feature)
        return graph_feature

    def each_forward(self, graph, node_feature):
        out = node_feature
        last_out = [node_feature]
        for i in range(self.depth):
            out = self.layers[i](graph, out)
            last_out.append(out)
        last_out = torch.cat(last_out,dim=-1)
        # import pdb ; pdb.set_trace()
        return last_out