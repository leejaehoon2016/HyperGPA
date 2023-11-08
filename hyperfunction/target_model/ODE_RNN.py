import torch
from torch import nn
from torchdiffeq import odeint
from .base import Base
from .base_model.Linear import MyLinear
from .base_model.GRU import MyGRUCell

# from base import Base
# from base_model.Linear import MyLinear
# from base_model.GRU import MyGRUCell


class Base_ODE_RNN(Base):
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_hidden_dim, num_layers, **kargs):
        super().__init__()
        self.in_size = in_dim
        in_dim = in_dim + 1
        self.hidden_dim = hidden_dim
        self.ode_cell = _ODERNNFunc(hidden_dim, hidden_hidden_dim, num_layers)
        self.cell = MyGRUCell(in_dim, hidden_dim)
        self.final_linear = MyLinear(hidden_dim, out_dim)
        self.option = {"method":"rk4", "rtol":1e-3, "atol":1e-3, "options" : {"step_size":1} }

    def forward(self, x, not_reuse_param = True):
        time = torch.arange(0,x.size(1)).float().to(x.device)
        t_tmp = time[None,:,None].repeat(x.size(0),1,1).to(x.device)
        x = torch.cat([t_tmp,x],dim=-1)
        h_i = torch.zeros(x.size(0), self.hidden_dim).to(x.device)

        # Get the odeint function
        ode_func = odeint

        # Loop over time to get the final hidden state
        dts = [torch.Tensor([t0, t1]).to(x.device) for t0, t1 in zip(time[:-1], time[1:])]
        for i in range(1,x.size(1)):
            # Solve ODE then update with data
            h_i = ode_func(func=self.ode_cell, y0=h_i, t=dts[i-1], **self.option)[-1]
            h_i = self.cell(x[:, i], h_i)

        outputs = self.final_linear(h_i) 
        
        if not_reuse_param:
            self.ode_cell.del_all_param()
            self.cell.del_all_param()
            self.final_linear.del_all_param()
        return outputs
    def init_param(self, dic):
        self.ode_cell.init_param({k[9:]:v for k,v in dic.items() if "ode_cell." == k[:9]})
        self.cell.init_param({k[5:]:v for k,v in dic.items() if "cell." == k[:5]})
        self.final_linear.init_param({k[13:]:v for k,v in dic.items() if "final_linear." == k[:13]})

    def get_computation_graph(self, complete_graph, connect_add, remove_selfloop, return_di):
        return self._get_computation_graph(torch.randn(5,2,self.in_size), complete_graph, connect_add, remove_selfloop, return_di)



class _ODERNNFunc(nn.Module):
    def __init__(self, hidden_dim, hidden_hidden_dim, num_layers):
        super().__init__()
        layer = nn.ModuleList()
        in_dim = hidden_dim
        mid_dims = [hidden_hidden_dim] * num_layers
        for next_dim in mid_dims:
            layer.append(MyLinear(in_dim, next_dim))
            # layer.append(nn.ReLU())
            in_dim = next_dim
        self.layers = layer
        self.last_layer = MyLinear(in_dim, hidden_dim)

    def forward(self,t,x):
        out = x
        for m in self.layers:
            out = m(out)
            out = out.relu()
        out = self.last_layer(out)
        return out
        
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
    model = Base_ODE_RNN(1,2,32,32,2)
    graph = model.get_computation_graph(0,0,1,1)
    import pdb ; pdb.set_trace()




