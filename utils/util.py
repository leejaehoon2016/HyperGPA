import os, json, copy, time, logging, random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn
import torch
import numpy as np

class IndexRandomGenerator:
    def __init__(self, max_len, num_iter_per_epoch, random = True):
        self.max_len = max_len
        self.batch   = max_len // num_iter_per_epoch + (max_len % num_iter_per_epoch != 0)
        self.random  = random
    def init(self):        
        self.indices = torch.randperm(self.max_len) if random else torch.arange(0,self.max_len)
        self.now_index = 0
        self.pre_index = None

    def next(self):
        ret = self.indices[self.now_index:self.now_index + self.batch]
        self.pre_index = self.now_index
        self.now_index = self.now_index + self.batch
        return ret
    def rep(self):
        ret = self.indices[self.pre_index:self.pre_index + self.batch]
        return ret


def fix_random_seed(random_num):
    torch.manual_seed(random_num)
    torch.cuda.manual_seed(random_num)
    torch.cuda.manual_seed_all(random_num) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_num)
    random.seed(random_num)


def save_score_model(val_dic, test_dic, model_state_dict, saved_dict, model_dict, epoch):
    change = False
    for name, val in val_dic.items():
        if name not in saved_dict:
            saved_dict[name] = (val, test_dic, epoch)
            model_dict["model"][name] = copy.deepcopy(model_state_dict)
            change = True
        elif saved_dict[name][0] < val:
            saved_dict[name] = (val, test_dic, epoch)
            model_dict["model"][name] = copy.deepcopy(model_state_dict)
            change = True
    return change, saved_dict, model_dict
def save_dict(path, dic):
    with open(path , 'w', encoding='utf-8') as f:
        json.dump(dic, f, indent="\t")

def mkdir(save_loc, d_folder, d_name, version):
    os.makedirs(f"{save_loc}/{d_folder}_{d_name}", exist_ok=True)
    if type(version) is int:
        version_name = f"version_{version}"
    elif type(version) is str:
        version_name = version
    else:
        used_num = [-1]
        for i in os.listdir(f"{save_loc}/{d_folder}_{d_name}"):
            try:
                used_num.append(int(i.split("_")[1]))
            except:
                pass
        version_num = max(used_num) + 1
        version_name = f"version_{version_num}"
    base = f"{save_loc}/{d_folder}_{d_name}/{version_name}"
    os.makedirs(base, exist_ok=True)
    os.makedirs(f"{base}/model", exist_ok=True)
    os.makedirs(f"{base}/score_info", exist_ok=True)
    return base

def cal_num_param(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def print_node_shape(model):
    model_dic = dict(model.state_dict())
    true_shape = {k:np.prod(v.shape) for k,v in model_dic.items()}
    return f"each_part_size: {true_shape}"

def get_PATH(dir_name):
    if dir_name != 'default':
        return './result/' + dir_name + '/'
    current_time = time.localtime()
    mon = '0' + str(current_time.tm_mon) if current_time.tm_mon < 10 else str(current_time.tm_mon)
    day = '0' + str(current_time.tm_mday) if current_time.tm_mday < 10 else str(current_time.tm_mday)
    current_time = f'{mon}{day} {current_time.tm_hour}:{current_time.tm_min}:{current_time.tm_sec}'
    return './result/' + current_time + '/'


def get_logger(PATH):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s      %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(PATH + '/train.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger