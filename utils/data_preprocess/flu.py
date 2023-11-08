import pandas as pd
import numpy as np
import torch
from .common import *
region1 = []
region1.append(["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"])
region1.append(["New Jersey", "New York", "New York City", "Puerto Rico", "U.S. Virgin Islands"])
region1.append(["Delaware", "District of Columbia", "Maryland", "Pennsylvania", "Virginia", "West Virginia"])
region1.append(["Alabama", "Florida", "Georgia", "Kentucky", "Mississippi", "North Carolina", "South Carolina", "Tennessee"])
region1.append(["Illinois", "Indiana", "Michigan", "Minnesota", "Ohio", "Wisconsin"])
region1.append(["Arkansas", "Louisiana", "New Mexico", "Oklahoma", "Texas"])
region1.append(["Iowa", "Kansas", "Missouri", "Nebraska"])
region1.append(["Colorado", "Montana", "North Dakota", "South Dakota", "Utah", "Wyoming"])
region1.append(["Arizona", "California", "Hawaii", "Nevada"])
region1.append(["Alaska", "Idaho", "Oregon", "Washington"])



region2 = []
region2.append(['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont'])
region2.append(['New Jersey', 'New York', 'New York City', 'Pennsylvania', 'PuertoRico', 'U.S.VirginIslands'])
region2.append(['Indiana', 'Illinois', 'Michigan', 'Ohio', 'Wisconsin'])
region2.append(['Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota'])
region2.append(['Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia', 'West Virginia'])
region2.append(['Alabama', 'Kentucky', 'Mississippi', 'Tennessee'])
region2.append(['Arkansas', 'Louisiana', 'Oklahoma', 'Texas'])
region2.append(['Arizona', 'Colorado', 'Idaho', 'New Mexico', 'Montana', 'Utah', 'Nevada', 'Wyoming'])
region2.append(['Alaska', 'California', 'Hawaii', 'Oregon', 'Washington'])


# region2_tmp = []
# for i in region2:
#     region2_tmp.extend(i)


def get_flu_data_val(num_test_period, num_val_period, lstm_x_len, lstm_y_len, hyper_x_len, hyper_y_len, model_type, print_info = True, period_start = 0, period_end = None):
    df = pd.read_csv(f"../data/flu/ILINet.csv", header=1)
    df = df.loc[:,(df == "X").sum() < 30000]
    df = df[(df == "X").sum(axis = 1) == 0]
    df[['YEAR', 'WEEK', 'ILITOTAL', 'NUM. OF PROVIDERS', 'TOTAL PATIENTS']] = df[['YEAR', 'WEEK', 'ILITOTAL', 'NUM. OF PROVIDERS', 'TOTAL PATIENTS']].astype(int)
    df['%UNWEIGHTED ILI'] = df['%UNWEIGHTED ILI'].astype(float)
    df = df.sort_values(["REGION", "YEAR","WEEK"])
    tmp = df.groupby(["REGION"]).size()
    tmp = tmp[tmp < 564]
    drop_state = tmp.index
    df = df[~df["REGION"].isin(drop_state)]
    df = df[~df["YEAR"].isin([2010, 2021])]
    col = ["ILITOTAL", "NUM. OF PROVIDERS", "TOTAL PATIENTS"]
    df[col] = (df[col] - df[col].mean()) / df[col].std()
    df = df.reset_index(drop = True)
    states_list = list(df["REGION"].unique())
    year_list = list(df["YEAR"].unique())

    if period_end is None:
        year_list = year_list[period_start:]
    else:
        year_list = year_list[period_start:period_end]

    total_period_num = len(year_list)
    data = [np.stack([df.loc[(df["REGION"] == state) & (df["YEAR"] == year),col].values for state in states_list]) \
            for year in year_list] 
    each_year_len = [i.shape[1] for i in data]

    if model_type != "base":
        assert num_val_period == 1, f"num_val_period != 1"
        # assert hyper_x_len >= 2, f"hyper_x_len({hyper_x_len}) >= 2"
        assert total_period_num >= (num_val_period+num_test_period), f"total_period_num >= num_val_period+num_test_period, {total_period_num} >= {num_val_period+num_test_period}"
        train_period_num = total_period_num - (num_val_period+num_test_period)
        assert train_period_num >= (hyper_x_len + hyper_y_len), f"train_period_num >= (hyper_x_len + hyper_y_len), {train_period_num} >= ({hyper_x_len} + {hyper_y_len})"
    min_lstm_data_len = min(each_year_len)
    assert min_lstm_data_len >= (lstm_x_len + lstm_y_len), f"min_lstm_data_len >= (lstm_x_len + lstm_y_len), {min_lstm_data_len} >= ({lstm_x_len} + {lstm_y_len})"

    
    test_data = np.concatenate(data[-num_test_period:],axis=1)
    val_data = np.concatenate(data[-(num_test_period+num_val_period):-num_test_period],axis=1)
    train_data = data[:-(num_test_period+num_val_period)]
    
    
    states_dict = dict([(v,i) for i,v in enumerate(states_list)])
    
    n_region1 = [[states_dict[j] for j in i if j in states_dict] for i in region1]
    n_region2 = [[states_dict[j] for j in i if j in states_dict] for i in region2]
    region_list = {1:n_region1, 2:n_region2}
    


    if model_type == "original":
        train_data = np.concatenate(train_data[:], axis=1)
        train_data = torch.tensor(train_data, dtype=torch.float64)
        val_data   = torch.tensor(val_data, dtype=torch.float64)
        test_data  = torch.tensor(test_data, dtype=torch.float64)
        return train_data, val_data, test_data
    elif model_type == "base":
        test_data = make_lstm_data(np.concatenate([val_data[:,-lstm_x_len:], test_data], axis=1), lstm_x_len, lstm_y_len)
        val_data = make_lstm_data(np.concatenate([train_data[-1][:,-lstm_x_len:], val_data], axis=1), lstm_x_len, lstm_y_len)
        train_data =  make_lstm_data(np.concatenate(train_data[:], axis=1), lstm_x_len, lstm_y_len)
        # train_data =  make_lstm_data(np.concatenate(train_data[2:], axis=1), lstm_x_len, lstm_y_len)
        if print_info:
            print_info_func(model_type, states_list, total_period_num, each_year_len, train_data, val_data, test_data)
        train_data = [torch.tensor(i, dtype=torch.float32) for i in train_data]
        val_data = [torch.tensor(i, dtype=torch.float32) for i in val_data]
        test_data = [torch.tensor(i, dtype=torch.float32) for i in test_data]
        return train_data, val_data, test_data
    elif model_type == "base_each":
        test_data = make_lstm_data(np.concatenate([val_data[:,-lstm_x_len:], test_data], axis=1), lstm_x_len, lstm_y_len, True)
        val_data = make_lstm_data(np.concatenate([train_data[-1][:,-lstm_x_len:], val_data], axis=1), lstm_x_len, lstm_y_len, True)
        train_data =  make_lstm_data(np.concatenate(train_data[:], axis=1), lstm_x_len, lstm_y_len, True)
        if print_info:
            print_info_func(model_type, states_list, total_period_num, each_year_len, train_data, val_data, test_data)
        train_data = [torch.tensor(i, dtype=torch.float32) for i in train_data]
        val_data = [torch.tensor(i, dtype=torch.float32) for i in val_data]
        test_data = [torch.tensor(i, dtype=torch.float32) for i in test_data]
        return train_data, val_data, test_data
    elif model_type == "base_trick":
        train_long = torch.tensor(np.concatenate(train_data[:], axis=1), dtype=torch.float32)
        test_data = make_lstm_data(np.concatenate([val_data[:,-lstm_x_len:], test_data], axis=1), lstm_x_len, lstm_y_len, True)
        val_data = make_lstm_data(np.concatenate([train_data[-1][:,-lstm_x_len:], val_data], axis=1), lstm_x_len, lstm_y_len, True)
        train_data =  make_lstm_data(np.concatenate(train_data[:], axis=1), lstm_x_len, lstm_y_len, True)
        if print_info:
            print_info_func(model_type, states_list, total_period_num, each_year_len, train_data, val_data, test_data)
        train_data = [torch.tensor(i, dtype=torch.float32) for i in train_data]
        val_data = [torch.tensor(i, dtype=torch.float32) for i in val_data]
        test_data = [torch.tensor(i, dtype=torch.float32) for i in test_data]
        return train_data, val_data, test_data, train_long
    else:
        train_data_lst = []
        for i in range(train_period_num - (hyper_x_len + hyper_y_len) + 1):
            tmp_x = np.concatenate(train_data[i : hyper_x_len + i],axis=1)
            tmp_y = np.concatenate([train_data[hyper_x_len+i-1][:,-lstm_x_len:]] + train_data[hyper_x_len + i : hyper_x_len + hyper_y_len + i],axis=1)
            train_data_lst.append([tmp_x,tmp_y])
        
        test_data = make_lstm_data(np.concatenate([val_data[:,-lstm_x_len:], test_data], axis=1), lstm_x_len, lstm_y_len, True)
        # test_data = [[np.concatenate(train_data[ -hyper_x_len + 1 : ] + [val_data],axis=1), test_data]]
        if hyper_x_len == 1:
            test_data = [[np.concatenate([val_data],axis=1), test_data]]
        else:
            test_data = [[np.concatenate(train_data[ -hyper_x_len + 1 : ] + [val_data],axis=1), test_data]]

        val_data = make_lstm_data(np.concatenate([train_data[-1][:,-lstm_x_len:], val_data], axis=1), lstm_x_len, lstm_y_len, True)
        val_data = [[np.concatenate(train_data[ -hyper_x_len: ],axis=1), val_data]]
        train_data = [ (i[0], make_lstm_data(i[1], lstm_x_len, lstm_y_len, True)) for i in train_data_lst]

        train_data = [[torch.tensor(i, dtype=torch.float32), [torch.tensor(j[0], dtype=torch.float32), torch.tensor(j[1], dtype=torch.float32)]] for i,j in train_data]
        val_data   = [[torch.tensor(i, dtype=torch.float32), [torch.tensor(j[0], dtype=torch.float32), torch.tensor(j[1], dtype=torch.float32)]] for i,j in val_data]
        test_data  = [[torch.tensor(i, dtype=torch.float32), [torch.tensor(j[0], dtype=torch.float32), torch.tensor(j[1], dtype=torch.float32)]] for i,j in test_data]

        front, back = model_type.split("_")[1:]

        change_front, pathx_dim_per_state, change_back, back_concat_info, front_state, back_state = \
            change_front_back(front, back, len(states_list), region_list, test_data[0][0].size(-1))

        train_data = [[change_front(i), change_back(j)] for i,j in train_data]
        train_lstm_bacth_num = [j[-1] for i,j in train_data]
        train_data = [[i,j[:-1]] for i,j in train_data]
        val_data   = [[change_front(i), change_back(j)] for i,j in val_data]
        val_lstm_bacth_num = [j[-1] for i,j in val_data]
        val_data = [[i,j[:-1]] for i,j in val_data]
        test_data  = [[change_front(i), change_back(j)] for i,j in test_data]
        test_lstm_bacth_num = [j[-1] for i,j in test_data]
        test_data = [[i,j[:-1]] for i,j in test_data]

        if print_info:
            print_info_func(model_type, states_list, total_period_num, each_year_len, train_data, val_data, test_data)

        final_index = [i[0].shape[1] for i in train_data]
        max_index = max(final_index)
        train_data = [[torch.cat([i,torch.zeros_like(i)[:,:max_index - i.shape[1]]],dim=1),j]for i,j in train_data]

        state_edge_list = [[],[]]
        return train_data, val_data[0], test_data[0], pathx_dim_per_state, train_lstm_bacth_num, val_lstm_bacth_num[0], test_lstm_bacth_num[0], len(front_state), len(back_state), state_edge_list, final_index, back_concat_info
    