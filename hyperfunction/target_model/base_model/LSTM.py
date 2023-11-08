import torch
from torch import nn

class MyLSTM(nn.LSTM):
    def __init__(self, in_dim, hidden_dim, num_layers, batch_first, bidirectional):
        super().__init__(in_dim, hidden_dim, num_layers = num_layers, batch_first = batch_first, bidirectional = bidirectional)
        self.all_param_name = list(dict(self.named_parameters()).keys())

    def init_param(self, dic):
        try:
            self.del_all_param()
        except:
            pass
        for param_name in self.all_param_name:
            setattr(self, param_name, dic[param_name])
    
    def del_all_param(self):
        for param_name in self.all_param_name:
            delattr(self, param_name)

    def forward(self, x, h = None):
        result = super().forward(x,h)
        return result

        
    #     self.is_ode = is_ode
    #     # print(self.all_param_name)
    # def init_param(self, dic):
    #     try:
    #         self.del_all_param()
    #     except:
    #         pass
    #     for param_name in self.all_param_name:
    #         setattr(self, param_name, dic[param_name])
    
    # def del_all_param(self):
    #     for param_name in self.all_param_name:
    #         delattr(self, param_name)

    # def forward(self, x, not_reuse):
    #     result = super().forward(x)
    #     if not_reuse and not self.is_ode:
    #         # Avoid side effects in nn.Module
    #         self.del_all_param()
    #     return result

if __name__ == "__main__":
    model = MyLSTM(1, 1, 2, batch_first=1, bidirectional=1)
    
    # import pdb; 