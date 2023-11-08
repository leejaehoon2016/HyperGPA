from .GRU import Base_GRU
from .LSTM import Base_LSTM
from .NCDE import Base_NeuralCDE
from .ODE_RNN import Base_ODE_RNN
from .Seq2seq import Base_Geq2geq, Base_Seq2seq
"""
first in_size
second out_size
get_computation_graph should be defined
"""
def get_TargetModel(model_name):
    model_name = model_name.lower()
    model = None
    if model_name == "lstm":
        model = Base_LSTM
    elif model_name == 'gru':
        model = Base_GRU
    elif model_name == 'odernn':
        model = Base_ODE_RNN
    elif model_name == 'ncde':
        model = Base_NeuralCDE
    elif model_name == "seq2seq":
        model = Base_Seq2seq
    elif model_name == "geq2geq":
        model = Base_Geq2geq
    return model