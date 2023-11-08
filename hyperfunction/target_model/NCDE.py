import torch
import torch.nn as nn
from . import torchcde
from .base_model.Linear import MyLinear
from .base import Base

# import torchcde
# from base_model.Linear import MyLinear
# from base import Base


class Base_NeuralCDE(Base):
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_hidden_dim, num_layers, **kargs):
        super().__init__()
        self.in_size = in_dim
        self.init = MyLinear(in_dim + 1, hidden_dim)
        self.func = NCDE_func(in_dim + 1, hidden_dim, hidden_hidden_dim, num_layers)
        self.fc = MyLinear(hidden_dim, out_dim)
        self.option = {"method":"rk4", "rtol":1e-3, "atol":1e-3, "options" : {"step_size":1} }

    def forward(self,x, not_reuse_param = True):
        time = torch.arange(0,x.size(1)).float().to(x.device)
        t_tmp = time[None,:,None].repeat(x.size(0),1,1).to(x.device)
        coeffs = torchcde.natural_cubic_coeffs(torch.cat([t_tmp,x],dim=-1)).to(x.device)
        X = torchcde.CubicSpline(coeffs)
        z0 = self.init(X.evaluate(0))
        out = torchcde.cdeint(X,self.func,z0,t=time[[0,-1]], adjoint=False, **self.option)
        out = out[:,-1]
        out = self.fc(out)

        if not_reuse_param:
            self.init.del_all_param()
            self.func.del_all_param()
            self.fc.del_all_param()
        return out 

    def init_param(self, dic):
        self.init.init_param({k[5:]:v for k,v in dic.items() if "init." == k[:5]})
        self.func.init_param({k[5:]:v for k,v in dic.items() if "func." == k[:5]})
        self.fc.init_param({k[3:]:v for k,v in dic.items() if "fc." == k[:3]})

    def get_computation_graph(self, complete_graph, connect_add, remove_selfloop, return_di):
        return self._get_computation_graph(torch.randn(5,2,self.in_size), complete_graph, connect_add, remove_selfloop, return_di)


class NCDE_func(nn.Module):
    def __init__(self, data_feature, hidden_dim, hidden_hidden_dim, num_layers):
        super().__init__()
        self.data_feature = data_feature
        self.hidden_dim = hidden_dim
        layer = nn.ModuleList()
        in_dim = hidden_dim
        mid_dims = [hidden_hidden_dim] * num_layers
        for next_dim in mid_dims:
            layer.append(MyLinear(in_dim, next_dim))
            # layer.append(nn.ReLU())
            in_dim = next_dim
        self.layers = layer
        self.last_layer = MyLinear(in_dim, data_feature * hidden_dim)

    def forward(self, t, x):
        out = x
        for m in self.layers:
            out = m(out)
            out = out.relu()
        out = self.last_layer(out)
        return out.reshape(-1,self.hidden_dim,self.data_feature)
        
    def init_param(self, dic):
        for i,m in enumerate(self.layers):
            layer_name = f"layers.{i}."
            layer_name_l = len(layer_name)
            m.init_param({k[layer_name_l:]:v for k,v in dic.items() if layer_name == k[:layer_name_l]})
        self.last_layer.init_param({k[11:]:v for k,v in dic.items() if "last_layer." == k[:11]})
        
    def del_all_param(self):
        for m in self.layers:
            m.del_all_param()
        self.last_layer.del_all_param()
        


if __name__ == "__main__":
    model = Base_NeuralCDE(1,2,32,32,2)
    graph = model.get_computation_graph(0,0,1,1)
    import pdb ; pdb.set_trace()


