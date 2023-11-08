import argparse, json
from train import Trainer
import warnings ; warnings.filterwarnings("ignore")
from config import mapping, model_param, data_param

arg = argparse.ArgumentParser()

arg.add_argument("--data",type=str, default="china30")

arg.add_argument("--l_model",type=str, default="gru")
arg.add_argument("--l_h",type=int, default=16)
arg.add_argument("--l_l",type=int, default=1)

arg.add_argument("--emb_size",type=int,default=128)
arg.add_argument("--attn_dim",type=int,default=1024)
arg.add_argument("--hyper_depth",type=int,default=3)
arg.add_argument("--hyper_h",type=int,default=128)

arg.add_argument("--size",type=int,default=128)
arg.add_argument("--mid_h",type=int,default=256)
arg.add_argument("--w_k",type=int,default=256) 

arg.add_argument("--graph1",type=str,default="avwgcn")
arg.add_argument("--graph2",type=str,default="gat")
arg.add_argument("--len",type=int,default=2)
arg.add_argument("--s_len",type=int,default=-1)
arg.add_argument('--l2',type=float,default=1e-6)
arg.add_argument('--num_class',type=int,default=5)
arg.add_argument('--task2_ratio',type=float,default=0.01)
arg.add_argument('--prior',type=float,default=-1.)

arg.add_argument("--gpu",type=int,default=0)
arg.add_argument("--r",type=int, default=0)

arg.add_argument('-not_default', action='store_true')

arg = arg.parse_args()



data_hparam, arg = data_param(arg.data, arg)
if not arg.not_default:
    with open("hypergpa_hparam.json","r") as st_json:
        hypergpa_hparam = json.load(st_json)
    arg.emb_size, arg.attn_dim, arg.num_class, arg.task2_ratio, arg.prior  = hypergpa_hparam[arg.data][arg.l_model]
    # arg.l_h, arg.l_l = 16, 1
    # arg.graph1, arg.graph2 = "avwgcn", "gat"
    arg.len = data_hparam["hyper_x_len"]
else:
    with open("hypergpa_hparam.json","r") as st_json:
        hypergpa_hparam = json.load(st_json)
    _,_,_,_, arg.prior  = hypergpa_hparam[arg.data][arg.l_model]
    data_hparam["hyper_x_len"] = arg.len

if arg.s_len > 0:
    data_hparam["hyper_x_len"] = arg.s_len
    arg.len = arg.s_len



lstm_hparam = model_param(arg.l_model, arg.l_h, arg.l_l)

emb_cde_process_args = {"emb_dim": arg.emb_size, "emb_init_func":"lrl", "emb_init_karg":{"mid_dims":[32] * 2}, "emb_cdegraph_func":arg.graph1, "emb_cdegraph_karg" : mapping(arg.graph1, 32, 2), 
                        "emb_cdemap_func": "lt", "emb_cdemap_karg":{ "mid_dim" : 64}, "emb_prod_method":"matmul", "emb_kinetic":False, "emb_residual":False,  "emb_evolve_way" : "each_state" , "emb_div_samples":1}

emb_process_hparam = {"emb_process_type" : "cde", "emb_process_args" : emb_cde_process_args} 
lstm_cde_process_args = {"lstm_param_hdim": 128, "lstm_init_func":"l", "lstm_init_karg":{}, "lstm_cdegraph_func":"dagnn", 
                         "lstm_cdegraph_karg" : {"hidden_dim":32, "num_layers":2, "bidirectional":True, "agg_x":False, "agg": "mattn_h", "out_wx":True, "recurr":1},
                        "lstm_cdemap_func": "lt", "lstm_cdemap_karg":{ "mid_dim" : 128}, "lstm_prod_method":"matmul", "lstm_kinetic":False, "lstm_residual":False, 
                        "lstm_evolve_way":"each_state" , "lstm_div_samples":1}
# lstm_cde_process_args = None
lstm_process_hparam = {"lstm_process_type" : "no", "lstm_process_args" : lstm_cde_process_args}
cde_rest_process_hparam = {"include_timeinfo": False, "interpolation_method" : "cubic", "interpolation_eps":None, 
                           "solver_method":"rk4", "rtol":1e-3, "atol":1e-3, "step_size":1, "adjoint":True}
rest_process_hparam = {"rest_process_args": cde_rest_process_hparam}


hyper_model_hparam = {"start_hyper_dim": arg.attn_dim,
                      "start_hyper_func": "lrl", "start_hyper_karg":{"mid_dims":[32] * 2},
                      "mid_hyper_apply" : True,
                      "mid_init_func"  : "lrl", 
                      "mid_init_karg"  : {"mid_dims":[arg.size]*3}, 
                      "mid_hyper_func" : arg.graph2,
                      "mid_hyper_karg" : mapping(arg.graph2, arg.hyper_h, arg.hyper_depth),
                      "mid_hyper_dim" : arg.mid_h,
                      "last_hyper_weight_model": "lrl_db", 
                      "last_hyper_weight_param"  : {"mid_dims" : [arg.w_k] * 2 ,"drop":0., "batch":0},
                      "last_hyper_rest_model": "lrl_db",
                      "last_hyper_rest_param"  : {"mid_dims" : [arg.w_k] * 2,"drop":0., "batch":0}}


r_hparam = {"save_loc" : "result", "test_name" : f"{arg.data}_{arg.l_model}_{arg.l_h}_{arg.l_l}_{arg.emb_size}_{arg.attn_dim}_{arg.graph1}_{arg.graph2}_{arg.len}_{arg.num_class}_{arg.task2_ratio}_{arg.prior}^{arg.r}",
            "lr" : arg.lr, "weight_decay" :arg.l2, "loss_func": "mse", 
            "path_batch_size": arg.path_batch_size, "max_batch_size": arg.max_batch_size, "epochs" : arg.epoch, "GPU_NUM" : arg.gpu, 
            "seed": arg.r, "e_ki_sq":0.1, "e_ki_quad":0.1, "z_ki_sq":0.1, "z_ki_quad":0.1, "num_time_model": arg.num_class, "task2_ratio" : arg.task2_ratio, "prior":arg.prior}

model = Trainer(lstm_hparam, emb_process_hparam, lstm_process_hparam, rest_process_hparam, hyper_model_hparam, data_hparam, r_hparam)
model.train()

