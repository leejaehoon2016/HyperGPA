from .map_func import LRL, LTLL, LT, L, LTL

# get (...,*input_shape) return (..., *output_shape)

def load_map_func(name):
    name = name.lower()
    model_class = None
    if name == "lrl":
        model_class = LRL
    elif name == "ltll":
        model_class = LTLL
    elif name == "l":
        model_class = L
    elif name == "lt":
        model_class = LT
    elif name == 'ltl':
        model_class = LTL
    return model_class