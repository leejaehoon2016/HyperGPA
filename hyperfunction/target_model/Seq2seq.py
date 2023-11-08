import torch
import torch.nn as nn
from .base_model.LSTM import MyLSTM
from .base_model.GRU import MyGRU
from .base_model.Linear import MyLinear
from .base import Base

# from base_model.LSTM import MyLSTM
# from base_model.Linear import MyLinear
# from base import Base

class Base_Seq2seq(Base):
    def __init__(self, in_dim, out_dim, **kargs):
        super().__init__()
        self.in_size = in_dim
        self.y_length = out_dim // in_dim
        self.encoder = MyLSTM(in_dim, kargs['hidden_dim'], kargs['num_layers'], batch_first = True, bidirectional = False)
        self.decoder = MyLSTM(in_dim, kargs['hidden_dim'], kargs['num_layers'], batch_first = True, bidirectional = False)
        self.decoder_fc = MyLinear(kargs['hidden_dim'], in_dim)

    def forward(self, x, not_reuse_param = True, y = None):
        if y != None:
            rst = self.teacher_forcing(x, y)
            if not_reuse_param : self.del_all_param()
            return rst
    # def forward(self, x, not_reuse_param = True):    
        _, (h, c) = self.encoder(x)

        x = x[:, [-1], :] ## x = (batch, 1, in_dim)
        rst = []
        for i in range(self.y_length):
            x, (h, c) = self.decoder(x, (h, c))
            x = self.decoder_fc(x) ## x = (batch, 1, in_dim)
            rst.append(x)
        rst = torch.cat(rst, dim = 1) ## rst = (batch, y_length, in_dim)
        rst = torch.flatten(rst, start_dim = 1) ## rst = (batch, out_dim)
        if not_reuse_param : self.del_all_param()
        return rst

    def teacher_forcing(self, x, y):
        _, (h, c) = self.encoder(x)
        x = x[:, [-1], :] ## x = (batch, 1, in_dim)
        y = y.view(y.shape[0], self.y_length, -1) ## y = (batch, y_length, in_dim)
        x = torch.cat([x, y], dim = 1) ## x = (batch, 1 + y_length, in_dim)

        rst = []
        for i in range(self.y_length):
            out, (h, c) = self.decoder(x[:, [i], :], (h, c))
            out = self.decoder_fc(out)
            rst.append(out)
        rst = torch.cat(rst, dim = 1) ## rst = (batch, y_length, in_dim)
        rst = torch.flatten(rst, start_dim = 1) ## rst = (batch, out_dim)
        return rst

    def del_all_param(self):
        self.encoder.del_all_param()
        self.decoder.del_all_param()
        self.decoder_fc.del_all_param()

    def init_param(self, dic):
        self.encoder.init_param({k[8:]:v for k,v in dic.items() if "encoder." == k[:8]})
        self.decoder.init_param({k[8:]:v for k,v in dic.items() if "decoder." == k[:8]})
        self.decoder_fc.init_param({k[11:]:v for k,v in dic.items() if "decoder_fc." == k[:11]})

    def get_computation_graph(self, complete_graph, connect_add, remove_selfloop, return_di):
        return self._get_computation_graph(torch.randn(5,5,self.in_size), complete_graph, connect_add, remove_selfloop, return_di)

class Base_Geq2geq(Base):
    def __init__(self, in_dim, out_dim, **kargs):
        super().__init__()
        self.in_size = in_dim
        self.y_length = out_dim // in_dim
        self.encoder = MyGRU(in_dim, kargs['hidden_dim'], kargs['num_layers'], batch_first = True, bidirectional = False)
        self.decoder = MyGRU(in_dim, kargs['hidden_dim'], kargs['num_layers'], batch_first = True, bidirectional = False)
        self.decoder_fc = MyLinear(kargs['hidden_dim'], in_dim)

    def forward(self, x, not_reuse_param = True, y = None):
        if y != None:
            rst = self.teacher_forcing(x, y)
            if not_reuse_param : self.del_all_param()
            return rst
    # def forward(self, x, not_reuse_param = True):    
        _, h = self.encoder(x)

        x = x[:, [-1], :] ## x = (batch, 1, in_dim)
        rst = []
        for i in range(self.y_length):
            x, h = self.decoder(x, h)
            x = self.decoder_fc(x) ## x = (batch, 1, in_dim)
            rst.append(x)
        rst = torch.cat(rst, dim = 1) ## rst = (batch, y_length, in_dim)
        rst = torch.flatten(rst, start_dim = 1) ## rst = (batch, out_dim)
        if not_reuse_param : self.del_all_param()
        return rst

    def teacher_forcing(self, x, y):
        _, h = self.encoder(x)
        x = x[:, [-1], :] ## x = (batch, 1, in_dim)
        y = y.view(y.shape[0], self.y_length, -1) ## y = (batch, y_length, in_dim)
        x = torch.cat([x, y], dim = 1) ## x = (batch, 1 + y_length, in_dim)

        rst = []
        for i in range(self.y_length):
            out, h = self.decoder(x[:, [i], :], h)
            out = self.decoder_fc(out)
            rst.append(out)
        rst = torch.cat(rst, dim = 1) ## rst = (batch, y_length, in_dim)
        rst = torch.flatten(rst, start_dim = 1) ## rst = (batch, out_dim)
        return rst

    def del_all_param(self):
        self.encoder.del_all_param()
        self.decoder.del_all_param()
        self.decoder_fc.del_all_param()

    def init_param(self, dic):
        self.encoder.init_param({k[8:]:v for k,v in dic.items() if "encoder." == k[:8]})
        self.decoder.init_param({k[8:]:v for k,v in dic.items() if "decoder." == k[:8]})
        self.decoder_fc.init_param({k[11:]:v for k,v in dic.items() if "decoder_fc." == k[:11]})

    def get_computation_graph(self, complete_graph, connect_add, remove_selfloop, return_di):
        return self._get_computation_graph(torch.randn(5,5,self.in_size), complete_graph, connect_add, remove_selfloop, return_di)

if __name__ == "__main__":
    karg = {'hidden_dim':16, 'num_layers':1}
    model = Base_Geq2geq(3,6, **karg)
    graph = model.get_computation_graph(0,0,1,1)
    import pdb ; pdb.set_trace()
