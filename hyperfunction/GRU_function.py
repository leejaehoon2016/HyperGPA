import imp
from re import I
from numpy import matmul
import torch
from torch.utils import data
import controldiffeq
from .linear_mapping.get_model import load_map_func
from .graph.get_model import load_graph_func
import dgl
from torch.nn import GRUCell, LSTMCell

class GraphGRU(torch.nn.Module):
    def __init__(self, device,
        include_timeinfo, interpolation_method, interpolation_eps, solver_method, rtol, atol, step_size, adjoint, 
        data_dim, lstm_node_dims, data_edge, lstm_edge, num_states,
        emb_dim, emb_init_func, emb_init_karg, emb_cdegraph_func, emb_cdegraph_karg, emb_cdemap_func, emb_cdemap_karg,
        emb_prod_method, emb_kinetic, emb_residual, emb_evolve_way, emb_div_samples,
        is_coevolving, 
        lstm_param_hdim = None, lstm_init_func = None, lstm_init_karg = None, lstm_cdegraph_func = None, lstm_cdegraph_karg = None, 
        lstm_cdemap_func = None, lstm_cdemap_karg = None, lstm_prod_method = None, lstm_kinetic = None, lstm_residual = None, 
        lstm_evolve_way = None, lstm_div_samples = None):
        # data_dim 어떤 경우에 +1

        super().__init__()
        self.include_timeinfo = include_timeinfo
        self.is_coevolving = is_coevolving
        self.num_states = num_states
        self.num_node = len(lstm_node_dims)
        self.interpolation_method = interpolation_method
        self.device = device
        self.emb_cdeint_kargs  = dict(zip(["emb_prod_method", "emb_kinetic", "emb_residual", "emb_evolve_way", "emb_div_samples"],[emb_prod_method, emb_kinetic, emb_residual, emb_evolve_way, emb_div_samples]))
        self.lstm_cdeint_kargs = dict(zip(["lstm_prod_method", "lstm_kinetic", "lstm_residual", "lstm_evolve_way", "lstm_div_samples"],[lstm_prod_method, lstm_kinetic, lstm_residual, lstm_evolve_way, lstm_div_samples]))
        self.origin_emb_kinetic = emb_kinetic
        self.origin_lstm_kinetic = lstm_kinetic
        
        assert emb_evolve_way != "concat", "emb_evolve_way should not be concat"

        if "recti" in self.interpolation_method or self.include_timeinfo:
            data_dim = [i+1  for i in data_dim]
        # if emb_evolve_way == "concat":
        #     data_dim = sum(data_dim)
        self.emb_init  = torch.nn.ModuleList()
        emb_init_func  = load_map_func(emb_init_func)
        self.data_dim = data_dim
        for i in data_dim:
            self.emb_init.append(emb_init_func(i, emb_dim,**emb_init_karg))
        
        self.emb_graph_func = graph_func(emb_cdegraph_func, emb_cdegraph_karg, emb_cdemap_func, emb_cdemap_karg, 
                                     emb_dim, data_dim, data_edge, num_states, emb_prod_method, device, "emb")
        
        # emb_cdemap_func = "lstm"
        grucell = LSTMCell if emb_cdemap_func == "lstm" else GRUCell
        print(grucell)
        self.grucells = torch.nn.ModuleList([ grucell(i,self.emb_graph_func.graph_output_dim) for i in data_dim])
        self.last_funcs = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.emb_graph_func.graph_output_dim, emb_dim), torch.nn.Tanh())
                            for i in data_dim])
        self.emb_cdemap_func = emb_cdemap_func
    def forward(self, x, final_index, return_type = "eT", return_kinetic = False):
        """
        x: (# batch, # state, length ,feature)
        return_type: ["eT","zT", "eAll"]
        """
        # import pdb ; pdb.set_trace()
        max_time = max(final_index)
        min_time = min(final_index)
        time = torch.arange(0,max_time).float().to(self.device)
        assert return_type in ["eT","zT", "eAll"], 'return_type in ["eT","zT", "eAll"]'
        assert not (return_type == "zT" and not self.is_coevolving), 'self.is_coevolving must be True, when return_type == "zT"'

        x0 = x[...,0,:]
        e0 = torch.stack([model(x0[:,i,:d]) for i, (d,model) in enumerate(zip(self.data_dim, self.emb_init))],dim=1) # (batch, state, feature)
        last = []
        if self.emb_cdemap_func == "lstm":
            e = e0
            c = None
            for time_i in range(max_time):
                graph_e = self.emb_graph_func(e)
                if c is None:
                    c = torch.zeros_like(graph_e).to(x.device)
                lstm_result = [each_model(x[:,i,time_i],(graph_e[:,i],c[:,i])) for i,each_model in enumerate(self.grucells)]
                e = torch.stack([self.last_funcs[i](lstm_result[i][0])  for i,each_model in enumerate(self.grucells)],dim=1)
                c = torch.stack([lstm_result[i][1]  for i,each_model in enumerate(self.grucells)],dim=1)
                last.append(e)
            
        else:
            e = e0
            for time_i in range(max_time):
                graph_e = self.emb_graph_func(e)
                gru_result = [each_model(x[:,i,time_i],graph_e[:,i]) for i,each_model in enumerate(self.grucells)]
                e = torch.stack([self.last_funcs[i](gru_result[i])  for i,each_model in enumerate(self.grucells)],dim=1)

                
                last.append(e)
                
        last = torch.stack(last,dim=0)

        return last[final_index -1, torch.arange(0,len(final_index))], {}



            


    
class graph_func(torch.nn.Module):
    def __init__(self, cdegraph_func, cdegraph_karg, cdemap_func, cdemap_karg, input_dim, matmul_dim, edge_index, num_node, prod_method, device, types):
        super().__init__()
        assert types in ["emb", "lstm"], 'CDE_func type in ["emb", "lstm"]'
        self.type = types
        self.input_dim = input_dim
        self.matmul_dim = matmul_dim
        self.prod_method = prod_method
        self.num_node = num_node
        self.g = dgl.graph((torch.LongTensor(edge_index[0]), torch.LongTensor(edge_index[1])), num_nodes=num_node)
        self.g = dgl.add_self_loop(self.g).to(device)
        graph_func = load_graph_func(cdegraph_func)
        self.graph_func = graph_func(input_dim, node_num=num_node, **cdegraph_karg) 
        self.graph_output_dim = self.graph_func.output_dim

    def forward(self, z_feature, dX_dts=None):
        """
        z_feature: (batch, num_states, num_node = None, node_feature)
        dX_dts:    (batch, num_states, num_node = None, node_feature) 
        """
        graph_feature = self.graph_func(self.g, z_feature) 
        return graph_feature

