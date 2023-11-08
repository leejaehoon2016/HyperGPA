
import torch
import numpy as np
def print_ilist_info(lst):
    result = []
    for e in lst:
        # print(type(e))
        if type(e) is not list and type(e) is not tuple:
            result.append(e.shape)
        else:
            result.append(print_ilist_info(e))
    return result

def make_lstm_data(data, lstm_x_len, lstm_y_len, retrain_state = False):
    """data: (state, len, feature)"""
    data =  data.transpose((1,0,2))
    length = len(data)
    x_data = [data[i : lstm_x_len + i] for i in range(length - (lstm_x_len + lstm_y_len) + 1)]
    y_data = [data[lstm_x_len + i : lstm_x_len + lstm_y_len + i] for i in range(length - (lstm_x_len + lstm_y_len) + 1)]
    if retrain_state:
        x_data, y_data = np.stack(x_data), np.stack(y_data) # (batch, len, state, feature)
        x_data = x_data.transpose((0,2,1,3))
        # import pdb ; pdb.set_trace()
        y_data = np.concatenate([y_data[:,i] for i in range(y_data.shape[1])],axis=-1)
    else:
        x_data, y_data = np.concatenate(x_data,axis=1), np.concatenate(y_data,axis=1)
        x_data = x_data.transpose((1,0,2))
        y_data = np.concatenate([y_data[i] for i in range(len(y_data))],axis=-1)
    return x_data, y_data

def change_front_back(front, back, num_state, region_list, data_dim):
    if front == "all":
        pathx_dim_per_state = [num_state * data_dim]
        change_front = lambda x : x.transpose(0,1).flatten(start_dim=1).unsqueeze(0)
        front_state = [[i for i in range(num_state)]]
        
    elif front[:6] =="region":
        region = region_list[int(front[6:])]
        max_len = max([len(i) for i in region])
        def change_front(x, region = region, max_len = max_len, data_dim = data_dim):
            max_x_len = data_dim * max_len
            x = x.transpose(0,1)
            tmp = [ x[:,r].flatten(start_dim=1) for r in region]
            x = torch.stack([ torch.cat([i,torch.zeros(i.size(0),max_x_len-i.size(1))],dim=-1) for i in tmp],dim=0)
            return x
        front_state = region
        pathx_dim_per_state = [data_dim*len(i) for i in region]
    else:
        pathx_dim_per_state = [data_dim for i in range(num_state)]
        change_front = lambda x : x
        front_state = [[i] for i in range(num_state)]
    
    if back == "all":
        change_back = lambda x : [x[0].flatten(start_dim=0, end_dim=1).unsqueeze(1), x[1].flatten(start_dim=0, end_dim=1).unsqueeze(1), [x[0].size(0)*x[0].size(1)]]
        back_state = [[i for i in range(num_state)]]
    elif back[:6] =="region":
        region = region_list[int(back[6:])]
        max_len = max([len(i) for i in region])
        def change_back(x, region = region, max_len = max_len):
            x, y = x
            max_x_len = x.size(0) * max_len
            tmp = [ x[:,r].flatten(start_dim=0,end_dim=1) for r in region]
            x = torch.stack([ torch.cat([i,torch.zeros(max_x_len-i.size(0), i.size(1), i.size(2))],dim=0) for i in tmp],dim=1)

            tmp = [ y[:,r].flatten(start_dim=0,end_dim=1) for r in region]
            y = torch.stack([ torch.cat([i,torch.zeros(max_x_len-i.size(0), i.size(1))],dim=0) for i in tmp],dim=1)

            return [x, y, [len(i) for i in tmp]]
        back_state = region
    else:
        change_back = lambda x : [x[0], x[1], [x[0].size(0) for i in range(x[0].size(1))]]
        back_state = [[i] for i in range(num_state)]
    
    
    new_back_state = []
    for bi, each_old_back_state in enumerate(back_state):
        new_back_state.append([])
        each_old_back_state = set(each_old_back_state)
        for fi, each_old_front_state in enumerate(front_state):
            if all(np.isin(each_old_front_state,list(each_old_back_state))):
                new_back_state[bi].append(fi)
                each_old_back_state = each_old_back_state - set(each_old_front_state)
        if each_old_back_state:
            assert 0, "Significant Error"



    return change_front, pathx_dim_per_state, change_back, new_back_state, front_state, back_state

def print_info_func(model_type, states_list, total_period_num, each_year_len, train_data, val_data, test_data):
    print(model_type)
    print("num of state:", len(states_list))
    print("total period num:",total_period_num)
    print("each period length:",each_year_len)
    print("train_data:", print_ilist_info(train_data))
    print("val_data: ",print_ilist_info(val_data))
    print("test_data:",print_ilist_info(test_data))