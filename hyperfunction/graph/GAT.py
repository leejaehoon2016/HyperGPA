import torch
import dgl
from torch import nn

"""
first arg have to be in_feats
output_dim attribute is needed
"""
class GAT(torch.nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, depth, mid_dim, mid_num_heads, **karg):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.depth = depth
        in_dim = in_feats
        # out_dim = 
        for i in range(depth-1):
            each_layer = dgl.nn.pytorch.conv.GATConv(in_dim, mid_dim, mid_num_heads)#,activation=nn.LeakyReLU(0.1),allow_zero_in_degree=True)
            self.layers.append(each_layer)
            in_dim = mid_dim * mid_num_heads
        self.layers.append(dgl.nn.pytorch.conv.GATConv(in_dim, out_feats, num_heads))#,activation=nn.LeakyReLU(0.1),allow_zero_in_degree=True))
        self.output_dim = in_feats + out_feats * num_heads * depth
        # self.output_dim = 
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

    # def each_forward(self, graph, node_feature):
    #     out = node_feature
    #     for i in range(self.depth):
    #         out = self.layers[i](graph, out)
    #         out = torch.flatten(out, start_dim=1)
    #     return out
    def each_forward(self, graph, node_feature):
        out = node_feature
        last_out = [node_feature]
        for i in range(self.depth):
            out = self.layers[i](graph, out)
            out = torch.flatten(out, start_dim=1)
            last_out.append(out)
        last_out = torch.cat(last_out,dim=-1)
        # import pdb ; pdb.set_trace()
        return last_out
"""
first arg have to be in_feats
output_dim attribute is needed
"""
class GAT_self(torch.nn.Module):
    def __init__(self, in_feats, node_num, out_feats, num_heads, depth, mid_dim, mid_num_heads, **karg):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.depth = depth
        in_dim = in_feats
        for i in range(depth-1):
            each_layer = dgl.nn.pytorch.conv.GATConv(in_dim, mid_dim, mid_num_heads,activation=nn.LeakyReLU(0.1),allow_zero_in_degree=True)# )
            self.layers.append(each_layer)
            in_dim = mid_dim * mid_num_heads
        self.layers.append(dgl.nn.pytorch.conv.GATConv(in_dim, out_feats, num_heads,allow_zero_in_degree=True))#,activation=nn.LeakyReLU(0.1),allow_zero_in_degree=True))
        self.output_dim = out_feats * num_heads
        self.main_graph = None
        self.main_batch = None
        self.map = nn.Linear(in_dim, self.output_dim)
        self.w   = nn.Linear(node_num * self.output_dim, node_num * self.output_dim) # nn.parameter.Parameter(torch.zeros(num_node, self.output_dim).float(), requires_grad=True)
        self.num_node = node_num
        
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
        origin_node_feature = node_feature
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
        
        w = self.w(graph_feature.flatten(start_dim=1)).view(-1,self.num_node,self.output_dim).sigmoid()
        graph_feature = (1-w) * graph_feature + w * self.map(origin_node_feature)
        return graph_feature

    def each_forward(self, graph, node_feature):
        out = node_feature
        for i in range(self.depth):
            out = self.layers[i](graph, out)
            out = torch.flatten(out, start_dim=1)
        return out
