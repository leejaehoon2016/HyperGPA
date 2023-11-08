from .empty import Empty
from .adaptive_graph import AVWGCN
from .GAT import GAT
from .GCN import GCN
def load_graph_func(name):
    name = name.lower()
    if name == "gat":
        model_class = GAT
    elif name == 'gcn':
        model_class = GCN
    elif name == 'empty':
        model_class = Empty
    elif name == "avwgcn":
        model_class = AVWGCN
    return model_class
