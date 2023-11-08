from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import torch.nn as nn
import torch
import numpy as np

def get_loss_function(prob_type):
    if prob_type == "regression":
        loss_func = nn.MSELoss()
    return loss_func

class RMSELoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


def cal_score(y_hat,y_true,t):
    if t == "regression":
        result = cal_regression(y_true, y_hat)
    return result
        

def cal_pearson(y_hat,y_true):
    vx = y_hat - y_hat.mean()
    vy = y_true - y_true.mean()
    cost = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return cost

def mean(lst):
    return float(sum(lst) / len(lst))

def cal_regression(y_true, y_hat):
    result = { i:[] for i in ["pearson", "r2", "explained_variance", "mean_squared", "mean_absolute", "mean_absolute_percentage_error"]}
    try:
        for i in range(y_hat.shape[-1]):
            y_h = y_hat[:,i]
            y_t = y_true[:,i]
            # import pdb ; pdb.set_trace()
            result["pearson"].append(cal_pearson(y_h,y_t))
            result["r2"].append(r2_score(y_h,y_t))
            result["explained_variance"].append(explained_variance_score(y_h,y_t))
            result["mean_squared"].append(-mean_squared_error(y_h,y_t))
            result["mean_absolute"].append(-mean_absolute_error(y_h,y_t))
            result["mean_absolute_percentage_error"].append(-mean_absolute_percentage_error(y_h,y_t))

        return {k:mean(v) for k,v in result.items()}
    except:
        return { i:-255. for i in ["pearson", "r2", "explained_variance", "mean_squared", "mean_absolute", "mean_absolute_percentage_error"]}
    # y_hat, y_true = y_hat.reshape(-1), y_true.reshape(-1)
    
    
