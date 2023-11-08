from .linear_generation import LRL, LRL_DB
# get (...,*input_shape) return (..., *output_shape)

def load_param_generation_func(name):
    name = name.lower()
    model_class = None
    if name == "lrl":
        model_class = LRL
    if name == "lrl_db":
        model_class = LRL_DB
    return model_class