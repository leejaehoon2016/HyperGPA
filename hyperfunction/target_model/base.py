import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchviz import make_dot
import networkx  as nx
import pydot 
import matplotlib.pyplot as plt

class Base(nn.Module):
    def __init__(self, ):
        super().__init__()
        # self.model = model

    def forward(self, x):
        pass
        # out = self.model(x)
        # return out
    
    def cal_num_param(self):
        return sum([np.product(i.shape) for i in self.model.parameters()])

    def _get_computation_graph(self, x, complete_graph, connect_add, remove_selfloop, return_di):
        model = self
        if complete_graph:
            param_node = [i for i in dict(model.named_parameters())]
            dic = dict(enumerate(param_node))
            edgelist = [[],[]]
            for src in range(len(param_node)):
                for dst in range(len(param_node)):
                    if remove_selfloop and src==dst:
                        continue
                    edgelist[0].append(src)
                    edgelist[1].append(dst)
            return dic, edgelist
        
        out = model(x,False)
        param_dic = dict(model.named_parameters())
        param_dic["output"] = out
        str_graph = str(make_dot(out, params=param_dic))
        graphs = pydot.graph_from_dot_data(str_graph)[0]
        G = nx.DiGraph(nx.nx_pydot.from_pydot(graphs))
        map_dict = {k : v.split("\n")[0].strip().strip('"') for k,v in G.nodes(data="label")}
        node_with_param_name = [i for i in param_dic.keys()]
        map_dict = {k: v  if v in node_with_param_name else f"{v}_{i}" for i,(k,v) in enumerate(map_dict.items())}
        G = nx.relabel_nodes(G, map_dict)
        nodes = [i for i in list(G.nodes) if del_node_candi(i, node_with_param_name)]
        for node in nodes:
            reverse_G = G.reverse(copy=True)
            out_neighbor = G[node]
            in_neighbor  = reverse_G[node]
            out_degree, in_degree = len(out_neighbor), len(in_neighbor)  
            G.remove_node(node)
            G.add_edges_from([(i,o) for o in out_neighbor for i in in_neighbor])

        nodes = list(G.nodes)
        for node in nodes:
            reverse_G = G.reverse(copy=True)
            out_neighbor = G[node]
            in_neighbor  = reverse_G[node]
            out_degree, in_degree = len(out_neighbor), len(in_neighbor)  
            if in_degree == 1 and out_degree > 0:
                G.remove_node(node)
                G.add_edges_from([(i,o) for o in out_neighbor for i in in_neighbor])

        param_node = [i for i in dict(model.named_parameters())]
        for node in param_node:
            for neighbor in G[node]:
                try:
                    G.nodes[neighbor]["effect_node"].append(node)
                except:
                    G.nodes[neighbor]["effect_node"] = [node]
            G.remove_node(node)
  
        include_node = [k for k,v in dict(G.nodes(data = "effect_node")).items() if v ]
        not_include_node = [k for k,v in dict(G.nodes(data = "effect_node")).items() if not v ]
        return_graph = nx.DiGraph()
        return_graph.add_nodes_from(param_node)

        for not_node in not_include_node:
            reverse_G = G.reverse(copy=True)
            out_neighbor = G[not_node]
            in_neighbor  = reverse_G[not_node]
            G.remove_node(not_node)
            G.add_edges_from([(i,o) for o in out_neighbor for i in in_neighbor])
        
        for src in include_node:
            # print(1)
            for dst in include_node:
                if not connect_add and "addback" in dst.lower():
                    continue
                connect = False
                if len(include_node) == len(G.nodes):
                    if src==dst:
                        continue
                    if (src,dst) in G.edges:
                        connect = True
                    else:
                        tmp_copy_G = G.copy()
                        for tmp_node_name in G.nodes:
                            if "addback" not in tmp_node_name.lower() and tmp_node_name != src and tmp_node_name != dst:
                                tmp_copy_G.remove_node(tmp_node_name)
                        if nx.has_path(tmp_copy_G, source=src, target=dst):
                            connect = True
                        
                else:
                    for path in nx.all_simple_paths(G, source=src, target=dst):
                        # import pdb ; pdb.set_trace()
                        result = [(i not in include_node) or ("addback" in i.lower()) for i in path[1:-1]]
                        if all(result):
                            connect = True
                            break
                if not connect:
                    continue
    
                src_node = G.nodes[src]["effect_node"]
                dst_node = G.nodes[dst]["effect_node"]
                if "addmmback" in dst.lower() and connect_add:
                    return_graph.add_edges_from([(s,d) for s in src_node for d in dst_node])
                elif "addmmback" in dst.lower() and not connect_add:
                    dst_node = [i for i in dst_node if ".bias" not in i]
                    return_graph.add_edges_from([(s,d) for s in src_node for d in dst_node])
                elif "mulback" in dst.lower() or "mmback" in dst.lower():
                    return_graph.add_edges_from([(s,d) for s in src_node for d in dst_node])
                elif "addback" in dst.lower() and connect_add:
                    return_graph.add_edges_from([(s,d) for s in src_node for d in dst_node])
        if remove_selfloop:
            return_graph.remove_edges_from(list(nx.selfloop_edges(return_graph)))
        if not return_di:
            # import pdb ; pdb.set_trace()
            return_graph = return_graph.to_undirected()
            return_graph = return_graph.to_directed()
        dic = dict(enumerate(return_graph.nodes))
        return_graph = nx.relabel_nodes(return_graph, {v:k for k,v in dic.items()})
        # import pdb; pdb.set_trace()
        return dic, [[i[0] for i in return_graph.edges], [i[1] for i in return_graph.edges]]

def check_no_param_in_neighbor(neighbor, node_with_param_name):
    result = [i not in node_with_param_name  for i in neighbor]
    # print(neighbor,all(result) if result else False)
    return all(result) if result else False

def del_node_candi(candi, node_with_param_name):
    result = True
    candi_ = candi.lower()
    if "addback" in candi_ or "mmback" in candi_ or "mulback" in candi_ or "addmmback" in candi_:
        result = False 
    if candi in node_with_param_name:
        result = False
    return result
            
        