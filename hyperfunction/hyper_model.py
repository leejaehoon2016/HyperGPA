import imp

from .CDE_function import GraphCDE
from .GRU_function import GraphGRU
import torch, dgl
import torch.nn as nn
from .param_generation.get_model import load_param_generation_func
from .target_model.get_model import get_TargetModel
from .graph.get_model import load_graph_func
from .linear_mapping.get_model import load_map_func

interpolation_methods = ["cubic", "linear", "rectilinear", "linear_cubic_smoothing", "linear_quintic_smoothing"]
prod_methods = ['evaluate', 'derivative','matmul']

def cal(model, attn, train, state):
    # idx = attn.argsort()[-3:]
    # model = model[...,idx]
    # attn  = attn[idx]
    # attn = attn/attn.sum()
        # stan = 1e-2
        # model = model[...,attn > ]
    #     plus = torch.zeros_like(attn)
    #     plus[attn < stan] = stan
    #     attn = attn + plus
    #     attn = attn/attn.sum()
    return model @ attn
    
    
    # self.model_param_lst[pi] @ attn[i][j][pi]

class HyperModel(torch.nn.Module):
    def __init__(self, path_data_dims, lstm_data_dim, lstm_out_dim, data_edge, front_num_states, back_num_states, concat_info_front2back, 
                 lstm_name, lstm_karg, lstm_graph_karg, num_batch, device, 
                 emb_process_type, emb_process_args, lstm_process_type, lstm_process_args, rest_process_args,
                 start_hyper_dim, start_hyper_func, start_hyper_karg, mid_hyper_apply, mid_init_func, mid_init_karg, mid_hyper_func, mid_hyper_karg, mid_hyper_dim,
                 last_hyper_weight_model, last_hyper_rest_model, last_hyper_weight_param, last_hyper_rest_param, num_time_model, prior_knowledge = -1):
                 
        super().__init__()
        self.num_time_model = num_time_model
        self.prior_knowledge =prior_knowledge
        lstm_class = get_TargetModel(lstm_name)
        lstm_model = lstm_class(lstm_data_dim, lstm_out_dim, **lstm_karg)
        self.idx2node_name, lstm_edge = lstm_model.get_computation_graph(**lstm_graph_karg)
        tmp_dic = { k : tuple(v.shape) for k,v in lstm_model.named_parameters()}
        self.lstm_node_dims   = [tmp_dic[self.idx2node_name[i]] for i in range(len(tmp_dic))]
        self.superficial_model_lst = [[lstm_class(lstm_data_dim, lstm_out_dim, **lstm_karg).to(device) for j in range(back_num_states)] for i in range(num_batch)]
        self.superficial_model_lst2 = [[lstm_class(lstm_data_dim, lstm_out_dim, **lstm_karg).to(device) for j in range(back_num_states)] for i in range(num_batch)]
        tmp_model = [dict(lstm_class(lstm_data_dim, lstm_out_dim, **lstm_karg).named_parameters()) for i in range(num_time_model)]
        self.model_param_lst = nn.ParameterList([nn.Parameter( torch.stack([tmp_model[i][k] for i in range(num_time_model)], dim=-1))  for k in tmp_dic])

        # if emb_process_type == "cde" and lstm_process_type == "cde":
        #     self.model_return_type = "zT"
        #     self.model_generation_start_dim = lstm_process_args["lstm_param_hdim"]
        #     self.model = GraphCDE(device=device, data_dim=data_dim, lstm_node_dims=self.lstm_node_dims, data_edge=data_edge,
        #              lstm_edge=lstm_edge, num_states=num_states, is_coevolving=True,  **emb_process_args, **lstm_process_args, **rest_process_args)

        if emb_process_type == "cde" and lstm_process_type == "no":
            self.model_return_type = "eT"
            self.model_generation_start_dim = emb_process_args["emb_dim"]
            self.model = GraphCDE(device=device, data_dim=path_data_dims, lstm_node_dims=self.lstm_node_dims, data_edge=data_edge,
                     lstm_edge=lstm_edge, num_states=front_num_states, is_coevolving=False,  **emb_process_args, **rest_process_args)
        elif emb_process_type == "cde" and lstm_process_type == "gru":
            raise NotImplementedError(f'emb_process_type{emb_process_type} or lstm_process_type{lstm_process_type} not implemented')
        elif emb_process_type == "gru" and lstm_process_type == "no":
            self.model_return_type = "eT"
            self.model_generation_start_dim = emb_process_args["emb_dim"]
            self.model = GraphGRU(device=device, data_dim=path_data_dims, lstm_node_dims=self.lstm_node_dims, data_edge=data_edge,
                     lstm_edge=lstm_edge, num_states=front_num_states, is_coevolving=False,  **emb_process_args, **rest_process_args)
        elif emb_process_type == "gru" and lstm_process_type == "gru":
            raise NotImplementedError(f'emb_process_type{emb_process_type} or lstm_process_type{lstm_process_type} not implemented')
        else:
            raise NotImplementedError(f'emb_process_type{emb_process_type} or lstm_process_type{lstm_process_type} not implemented')        
        
        num_node = len(self.idx2node_name)
        self.query_attn = nn.Linear(self.model_generation_start_dim, start_hyper_dim * num_node, bias=False)
        
        graph_func = load_graph_func(mid_hyper_func)
        self.graph_func = graph_func(start_hyper_dim, node_num=num_node, **mid_hyper_karg) 
        self.lstm_g = dgl.graph((torch.LongTensor(lstm_edge[0]), torch.LongTensor(lstm_edge[1])), num_nodes=num_node)
        self.lstm_g = dgl.add_self_loop(self.lstm_g).to(device)
        
        self.key_attn  = torch.nn.ModuleList()
        for i in range(num_node):
            x = nn.Linear(self.graph_func.output_dim, self.num_time_model, bias=False)
            ##########################
            if prior_knowledge >= 0:
                x.weight.data.fill_(0)
            ##########################
            self.key_attn.append(x)


        # # # start_hyper
        # # self.concat_info_front2back = concat_info_front2back
        # # self.start_hyper_list = nn.ModuleList()
        # # start_hyper_func = load_map_func(start_hyper_func)
        # # for each_info in concat_info_front2back:
        # #     self.start_hyper_list.append(start_hyper_func(len(each_info) * self.model_generation_start_dim, start_hyper_dim, **start_hyper_karg))
        
        # # mid_hyper
        # self.mid_hyper_apply = mid_hyper_apply
        # if mid_hyper_apply:
        #     num_node = len(self.idx2node_name)
        #     self.mid_hyper_init  = torch.nn.ModuleList()
        #     mid_init_func  = load_map_func(mid_init_func)
        #     for i in range(num_node):
        #         self.mid_hyper_init.append(mid_init_func(start_hyper_dim, mid_hyper_dim,**mid_init_karg))
            
        #     graph_func = load_graph_func(mid_hyper_func)
        #     self.graph_func = graph_func(mid_hyper_dim, node_num=num_node, **mid_hyper_karg) 
        #     self.lstm_g = dgl.graph((torch.LongTensor(lstm_edge[0]), torch.LongTensor(lstm_edge[1])), num_nodes=num_node)
        #     self.lstm_g = dgl.add_self_loop(self.lstm_g).to(device)

        #     start_hyper_dim = self.graph_func.output_dim

        # weigth_hyper_class = load_param_generation_func(last_hyper_weight_model)
        # bias_hyper_class = load_param_generation_func(last_hyper_rest_model)
        # self.hyper_model_list = nn.ModuleList()

        # for i in range(len(self.idx2node_name)):
        #     node_dim = self.lstm_node_dims[i]
        #     # import pdb ; pdb.set_trace()
        #     if len(node_dim) == 2:
        #         tmp = weigth_hyper_class(start_hyper_dim, node_dim, **last_hyper_weight_param)
        #     else:
        #         tmp = bias_hyper_class(start_hyper_dim, node_dim, **last_hyper_rest_param)
        #     self.hyper_model_list.append(tmp)

        
    def forward(self, x_path, final_index, return_kinetic = False, train = False):
        emb, reg = self.model(x_path, final_index, self.model_return_type, return_kinetic) # (batch, num_state, (num_node,) self.model_generation_start_dim)
        num_node = len(self.idx2node_name)

        query = self.query_attn(emb).chunk(num_node,dim=-1)
        query = torch.stack(query,dim=2) # (batch, num_state, num_node, self.model_generation_start_dim)
        num_batch, num_state = query.size(0), query.size(1)
        query = torch.cat([query[i] for i in range(len(query))],dim=0) 
        query = self.graph_func(self.lstm_g,query)
        query = torch.stack([query[i*num_state:(i+1)*num_state] for i in range(num_batch)],dim=0) # (batch, num_state, num_node, self.model_generation_start_dim)
        
        attn_o = torch.stack([self.key_attn[i](query[:,:,i]) for i in range(num_node)], dim=2) # (batch, num_state, num_node, self.model_generation_start_dim)
        if self.prior_knowledge > 0:
            attn_o = attn_o.sigmoid()
            tmp = torch.zeros_like(attn_o)
            le = attn_o.size(1)
            tmp[:,torch.arange(le),:,torch.arange(le)] = self.prior_knowledge
            attn_o = attn_o + tmp
        attn = torch.nn.functional.softmax(attn_o,dim=-1)

        num_batch, num_states = emb.size(0), emb.size(1)
        [[self.superficial_model_lst[i][j].init_param({k:cal(self.model_param_lst[pi], attn[i][j][pi], train, j)  for pi, k in self.idx2node_name.items()},) 
        for j in range(num_states)] for i in range(num_batch)]
        [[self.superficial_model_lst2[i][j].init_param({k:self.model_param_lst[pi][...,attn[i][j][pi].argmax()] for pi, k in self.idx2node_name.items()},) 
        for j in range(num_states)] for i in range(num_batch)]
        if train:
            return [[self.superficial_model_lst[i][j] for j in range(num_states)] for i in range(num_batch)], [[self.superficial_model_lst2[i][j] for j in range(num_states)] for i in range(num_batch)], attn, reg
        else:
            return [[self.superficial_model_lst[i][j] for j in range(num_states)] for i in range(num_batch)], reg
        # # :
        # #     attn.append()
        

        # # emb_ori = emb
        # # # start 
        # # if len(emb.size()) == 4:
        # #     emb = emb.transpose(1,2)
        
        # # emb = [each_start_hyper_model(emb[...,each_info,:].flatten(start_dim=-2)) \
        # #     for each_info, each_start_hyper_model in zip(self.concat_info_front2back, self.start_hyper_list)]
        # # emb = torch.stack(emb,dim=1)
        # # import pdb ; pdb.set_trace()


        # if self.mid_hyper_apply:
        #     emb_lst = []
        #     for i in range(len(self.idx2node_name)):
        #         if self.model_return_type == "eT":
        #             emb_lst.append(self.mid_hyper_init[i](emb))    
        #         elif self.model_return_type == "zT":
        #             emb_lst.append(self.mid_hyper_init[i](emb[:,:,i]))
        #     emb = torch.stack(emb_lst,dim=2) # (batch, num_state, num_node, self.model_generation_start_dim)
            
        #     num_batch, num_state = emb.size(0), emb.size(1)
        #     emb = torch.cat([emb[i] for i in range(len(emb))],dim=0) 
        #     emb = self.graph_func(self.lstm_g,emb)
        #     emb = torch.stack([emb[i*num_state:(i+1)*num_state] for i in range(num_batch)],dim=0) # (batch, num_state, num_node, self.model_generation_start_dim)

        # if self.model_return_type == "zT" or self.mid_hyper_apply:
        #     return_model_param = {node_name:self.hyper_model_list[idx](emb[:,:,idx]) for idx, node_name in self.idx2node_name.items()}

        # elif  self.model_return_type == "eT":
        #     return_model_param = {node_name:self.hyper_model_list[idx](emb) for idx, node_name in self.idx2node_name.items()}
        # num_batch, num_states = emb.size(0), emb.size(1)
        # [[self.lstm_model_lst[i][j].init_param({k:v[i][j] for k,v in return_model_param.items()},) for j in range(num_states)] for i in range(num_batch)]
        # if train_emb:
        #     return [[self.lstm_model_lst[i][j] for j in range(num_states)] for i in range(num_batch)], reg, emb_ori
        # return [[self.lstm_model_lst[i][j] for j in range(num_states)] for i in range(num_batch)], reg
    