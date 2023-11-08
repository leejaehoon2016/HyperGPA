def mapping(name, hdim, depth):
    name = name.lower()
    if name == "gcn":
        return {"out_feats": hdim, "depth":depth, "mid_dim":hdim}
    elif name == "gat":
        return {"out_feats": hdim, "num_heads": 4, "depth": depth, "mid_dim":hdim, "mid_num_heads" : 4}
    elif name == "avwgcn":
        return {"out_feats": hdim, "embed_dim": 32, "cheb_k":depth}
    elif name == "empty":
        return {}
    else:
        raise Exception


def model_param(model_type, h, l):
    model_type = model_type.lower()
    if model_type in ["lstm","gru"]:
        dic = {"name" : model_type, "karg" : {'hidden_dim' : h, 'num_layers':l, 'bidirectional':False}} 
    elif model_type in ["ncde","odernn"]:
        dic = {"name" : model_type, "karg" : {'hidden_dim' : h, 'hidden_hidden_dim': int(h * 2), 'num_layers' : l}}
    elif model_type in ["seq2seq","geq2geq"]:
        dic = {"name" : model_type, "karg" : {'hidden_dim' : h, 'num_layers':l}}
    dic = {"lstm_name": dic['name'], "lstm_karg": dic['karg'], "lstm_graph_karg": {"complete_graph":False, "connect_add":False, "remove_selfloop":True, "return_di":True,}}
    return dic

def data_param(d_folder, args):
    args.lr = 1e-2

    if d_folder == "flu":
        args.epoch = 150
        args.path_batch_size = 8
        args.max_batch_size  = 10000
        data_hparam = {"d_folder": "flu", "d_name":"ILINet", "num_test_period" : 1, "num_val_period" : 1, 
                       "model_type": "hyper_each_each",
                       "lstm_x_len":10, "lstm_y_len":2, "hyper_x_len":2, "hyper_y_len":1, "period_start": 0, "period_end": None}    
    elif d_folder == "ushcn":
        args.lr = 1e-2
        args.epoch = 300
        args.path_batch_size = 8
        args.max_batch_size  = 10000
        data_hparam = {"d_folder": "ushcn", "d_name":"ushcn", "num_test_period" : 1, "num_val_period" : 1, "model_type": "hyper_each_each",
                       "lstm_x_len":10, "lstm_y_len":2, "hyper_x_len":3, "hyper_y_len":1, "period_start": 0, "period_end": None}

    elif d_folder == "china30":
        args.epoch = 300
        args.path_batch_size = 8
        args.max_batch_size  = 10000
        data_hparam = {"d_folder": "stock", "d_name":"china30", "num_test_period" : 1, "num_val_period" : 1, "model_type": "hyper_each_each",
                       "lstm_x_len" : 10, "lstm_y_len" : 4, "hyper_x_len" : 2, "hyper_y_len":1, "period_start": 0, "period_end": None}
    
    elif d_folder in ["usa30"]:
        args.epoch = 300
        args.path_batch_size = 8
        args.max_batch_size  = 10000
        data_hparam = {"d_folder": "stock", "d_name":"usa30", "num_test_period" : 1, "num_val_period" : 1, "model_type": "hyper_each_each",
                       "lstm_x_len":10, "lstm_y_len":4, "hyper_x_len":2, "hyper_y_len":1, "period_start": 0, "period_end": None}
    return data_hparam, args
