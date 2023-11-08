import torch
import torch.nn as nn
from .base_model.GRU import MyGRU
from .base_model.Linear import MyLinear
from .base import Base

# from base_model.GRU import MyGRU
# from base_model.Linear import MyLinear
# from base import Base

"""
first arg: in_dim
second arg: out_dim
"""
class Base_GRU(Base):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, bidirectional = False, **kargs):
        super(Base_GRU, self).__init__()
        self.in_size = in_dim
        self.GRU = MyGRU(in_dim, hidden_dim, num_layers = num_layers, batch_first = True, bidirectional = bidirectional)
        self.fc = MyLinear(hidden_dim, out_dim)
        if bidirectional:
            hidden_dim *= 2

    def forward(self, x, not_reuse_param = True):
        out, _ = self.GRU(x)
        out = out[:,-1]
        out = self.fc(out)
        if not_reuse_param:
            self.GRU.del_all_param()
            self.fc.del_all_param()
        return out

    def init_param(self, dic):
        self.GRU.init_param({k[4:]:v for k,v in dic.items() if "GRU." == k[:4]})
        self.fc.init_param({k[3:]:v for k,v in dic.items() if "fc." == k[:3]})

    def get_computation_graph(self, complete_graph, connect_add, remove_selfloop, return_di):
        return self._get_computation_graph(torch.randn(5,5,self.in_size), complete_graph, connect_add, remove_selfloop, return_di)

if __name__ == "__main__":
    model = Base_GRU(1,2,32,2,False)
    graph = model.get_computation_graph(0,0,1,1)
    import pdb ; pdb.set_trace()
