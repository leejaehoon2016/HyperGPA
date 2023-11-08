import numpy as np
import random
from .data_preprocess.flu import get_flu_data_val
from .data_preprocess.ushcn import get_ushcn_data_val
from .data_preprocess.stock import get_stock_data_val

def make_hyper_choice(candi):
    hyper_choice = []
    for i,start_c in enumerate(candi[1:],1):
        for start in start_c:
            hyper_choice.append(f"hyper_{start}_{start}")
            for next_c in candi[:i]:
                for next in next_c:
                    hyper_choice.append(f"hyper_{start}_{next}")
    return hyper_choice


def get_data(d_folder, d_name, num_test_period, num_val_period, lstm_x_len, lstm_y_len, hyper_x_len, hyper_y_len, model_type, print_info = True, period_start=0, period_end=None):
    if d_folder == "flu" and  d_name == "ILINet":
        candi = [["all"],["region1","region2"],["each"]]
    else: 
        candi = [["all"],["each"]]
    hyper_choice = make_hyper_choice(candi)
    assert model_type in ["original", "base","base_toy","base_each","base_trick", "arima"] + hyper_choice, f"get_data : model_type({model_type}) in {hyper_choice}"
    
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    if d_folder == "flu" and  d_name == "ILINet": # and num_val_period:
        prob_type = "regression"
        data = get_flu_data_val(num_test_period, num_val_period, lstm_x_len, lstm_y_len, hyper_x_len, hyper_y_len, model_type, print_info, period_start, period_end)
    elif d_folder == "ushcn" and  d_name == "ushcn":
        prob_type = "regression"
        data = get_ushcn_data_val(num_test_period, num_val_period, lstm_x_len, lstm_y_len, hyper_x_len, hyper_y_len, model_type, print_info, period_start, period_end)
    elif d_folder == "stock":
        prob_type = "regression"
        data = get_stock_data_val(d_name, num_test_period, num_val_period, lstm_x_len, lstm_y_len, hyper_x_len, hyper_y_len, model_type, print_info, period_start, period_end)
    elif d_folder == "toy":
        prob_type = "regression"
        data = get_toy_data_val(d_name, num_test_period, num_val_period, lstm_x_len, lstm_y_len, hyper_x_len, hyper_y_len, model_type, print_info, period_start, period_end)
    return data, prob_type

# if __name__ == "__main__":
    


