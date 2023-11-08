import torch
from torch import nn


class MyLinear(nn.Linear):
    def __init__(self, in_size, out_size, bias = True):
        super().__init__(in_size, out_size, bias)
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

    def forward(self, x):
        result = super().forward(x)
        return result


# in_size = 10
# out_size = 20

# my_layer = MyLinear(in_size, out_size)

# inp = torch.rand(5, in_size)
# out = my_layer(inp)
# out.sum().backward()

# print(my_layer.M.grad)

if __name__ == "__main__":
    x = torch.rand(1,1)
    m = nn.Linear(1,1)
    w = m(x)
    b = w.reshape(1)
    my = MyLinear(1,1)
    my.init_param({"weight":  w, "bias" : b})
    import pdb ; pdb.set_trace()
