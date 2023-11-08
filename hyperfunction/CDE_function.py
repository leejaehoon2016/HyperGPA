from re import I
from numpy import matmul
import torch
from torch.utils import data
import controldiffeq
from .linear_mapping.get_model import load_map_func
from .graph.get_model import load_graph_func
import dgl

interpolation_methods = ["cubic", "linear", "rectilinear", "linear_cubic_smoothing", "linear_quintic_smoothing"]
prod_methods = ['evaluate', 'derivative','matmul']

def define_interpolation(interpolation_method, interpolation_eps):
    assert interpolation_method in interpolation_methods, f"interpolation_method should be {interpolation_methods}"
    if interpolation_method == "cubic":
        coeffs_func = controldiffeq.natural_cubic_coeffs
        spline = controldiffeq.NaturalCubicSpline
    elif interpolation_method == "linear":
        coeffs_func = controldiffeq.linear_interpolation_coeffs
        spline = controldiffeq.LinearInterpolation
    elif interpolation_method == "rectilinear":
        coeffs_func = controldiffeq.linear_interpolation_coeffs
        spline = controldiffeq.LinearInterpolation
    elif interpolation_method == "linear_cubic_smoothing":
        coeffs_func = controldiffeq.linear_interpolation_coeffs
        spline = lambda coeffs: controldiffeq.SmoothLinearInterpolation(
            coeffs,
            gradient_matching_eps=interpolation_eps,
            match_second_derivatives=False,
        )
    
    elif interpolation_method == "linear_quintic_smoothing":
        coeffs_func = controldiffeq.linear_interpolation_coeffs
        spline = lambda coeffs: controldiffeq.SmoothLinearInterpolation(
            coeffs,
            gradient_matching_eps=interpolation_eps,
            match_second_derivatives=True,
        )
    return coeffs_func, spline

def ode_args(solver_method, rtol, atol, step_size):
    solver_karg = {"method" : solver_method, "rtol":rtol, "atol":atol}
    if solver_method not in ["dopri8", "dopri5", "bosh3", "adaptive_heun", "sym12async"]:
        solver_karg["options"] = {'step_size': step_size}
    return solver_karg

class GraphCDE(torch.nn.Module):
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
        self.solver_kargs = ode_args(solver_method, rtol, atol, step_size) ; self.adjoint = adjoint
        self.coeffs_func, self.spline = define_interpolation(interpolation_method, interpolation_eps)
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

        self.emb_cde_func = CDE_func(emb_cdegraph_func, emb_cdegraph_karg, emb_cdemap_func, emb_cdemap_karg, 
                                     emb_dim, data_dim, data_edge, num_states, emb_prod_method, device, "emb")
        self.lstm_cde_func = None
        # if is_coevolving:
        #     for i in [lstm_param_hdim, lstm_init_func, lstm_init_karg, lstm_cdegraph_func, lstm_cdegraph_karg, lstm_cdemap_func, lstm_cdemap_karg]:
        #         assert i is not None, "if is_coevolving is true, you have to set [lstm_param_hdim, lstm_init_func, lstm_init_karg, lstm_cdegraph_func, lstm_cdegraph_karg, lstm_cdemap_func, lstm_cdemap_karg]"
        #     self.lstm_init  = torch.nn.ModuleList()
        #     lstm_init_func  = load_map_func(lstm_init_func)
        #     for i in lstm_node_dims:
        #         self.lstm_init.append(lstm_init_func(emb_dim,lstm_param_hdim,**lstm_init_karg))
        #     tmp_emb_dim = emb_dim
        #     if lstm_evolve_way == "concat":
        #         tmp_emb_dim = emb_dim * num_states
        #     self.lstm_cde_func = CDE_func(lstm_cdegraph_func, lstm_cdegraph_karg, lstm_cdemap_func, lstm_cdemap_karg, 
        #                                   lstm_param_hdim, tmp_emb_dim, lstm_edge, len(lstm_node_dims), lstm_prod_method, device, "lstm")

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
        # time = torch.arange(0,x.size(-2),dtype=int).float().to(self.device)
        if "recti" in self.interpolation_method:
            t_tmp = time[None,None,:,None].repeat(x.size(0),x.size(1),1,1).to(x.device)
            train_coeffs = self.coeffs_func(torch.cat([t_tmp,x],dim=-1),rectilinear=0).to(self.device)
        elif self.include_timeinfo:
            
            t_tmp = time[None,None,:,None].repeat(x.size(0),x.size(1),1,1).to(x.device) * 0.1
            train_coeffs = self.coeffs_func(torch.cat([t_tmp,x],dim=-1)).to(self.device)
        else:
            train_coeffs = self.coeffs_func(x).to(self.device)
        spline = self.spline(train_coeffs)
        x0 = spline.evaluate(0) # (batch, state, feature)

        e0 = torch.stack([model(x0[:,i,:d]) for i, (d,model) in enumerate(zip(self.data_dim, self.emb_init))],dim=1) # (batch, state, feature)
        if "recti" in self.interpolation_method:
            times = spline.grid_points
        else:
            times = spline.grid_points
        
        z0 = None
        # if self.is_coevolving and return_type == "zT":
        #     z0 = torch.stack([self.lstm_init[node_i](e0) for node_i in range(self.num_node)],dim=2) # (batch, state, num node, feature)
        #     times = times[[0] + list(range(min_time-1, max_time))]
        #     final_index = final_index - min_time + 1
        
        if return_type == "eT":
            times = times[[0] + list(range(min_time-1, max_time))]
            final_index = final_index - min_time + 1
        self.emb_cdeint_kargs["emb_kinetic"] = return_kinetic and self.origin_emb_kinetic
        # if self.is_coevolving and return_type == "zT":
        #     self.lstm_cdeint_kargs["lstm_kinetic"] = return_kinetic and self.origin_lstm_kinetic
        
        out_dic = controldiffeq.cdeint(e0=e0, emb_func=self.emb_cde_func, emb_X=spline, **self.emb_cdeint_kargs,
                                   z0=z0, lstm_func=self.lstm_cde_func, **self.lstm_cdeint_kargs,
                                   is_coevolving=self.is_coevolving and return_type == "zT", t=times, adjoint=self.adjoint, **self.solver_kargs)
        # import pdb ; pdb.set_trace()
        # ["e", "e_ki_sq", "e_ki_quad", "z", "z_ki_sq", "z_ki_quad"]
        reg = {k : v[-1].mean() for k,v in out_dic.items() if "ki" in k}
        if self.is_coevolving and return_type == "zT":
            zT = out_dic["z"][final_index, torch.arange(0,len(final_index))] # (batch, state, num node, feature)
            return zT, reg
        elif return_type == "eT":
            eT = out_dic["e"][final_index, torch.arange(0,len(final_index))] # (batch, state, feature)
            return eT, reg
        else:
            return out_dic["e"], reg # (length, batch, state, feature)

    
class CDE_func(torch.nn.Module):
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
        cdemap_func = load_map_func(cdemap_func)
        self.graph_last = torch.nn.ModuleList()
        self.matmul_dim = matmul_dim
        for i in matmul_dim:
            if prod_method == "matmul":
                self.graph_last.append(cdemap_func(self.graph_output_dim, input_dim * i, **cdemap_karg))
            else:
                self.graph_last.append(cdemap_func(self.graph_output_dim + i, input_dim, **cdemap_karg))
    def forward(self, z_feature, dX_dts=None):
        """
        z_feature: (batch, num_states, num_node = None, node_feature)
        dX_dts:    (batch, num_states, num_node = None, node_feature) 
        """
        if self.type == "emb":
            graph_feature = self.graph_func(self.g, z_feature) 
        else:
            batch_len = z_feature.size(0)
            state_len = z_feature.size(1)
            z_feature = torch.cat([z_feature[i] for i in range(batch_len)], dim = 0)
            graph_feature = self.graph_func(self.g, z_feature)# [self.graph_func(self.g, z_feature[])  for i in z_feature.size(1)]
            # import pdb ; pdb.set_trace()
            graph_feature = torch.stack([graph_feature[i*state_len:(i+1)*state_len] for i in range(batch_len)],dim=0)
        
        if self.prod_method == "matmul":
            graph_feature_shape = graph_feature.shape[:-2]
            result = [self.graph_last[i](graph_feature[...,i,:]).view(*graph_feature_shape, self.input_dim, d) for i,d in enumerate(self.matmul_dim)]
        else:
            result = [self.graph_last[i](torch.cat([graph_feature[...,i,:], dX_dts[...,i,:d]], dim=-1)) for i,d in enumerate(self.matmul_dim)]
        return result

