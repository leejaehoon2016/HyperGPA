import torch
import torchdiffeq
import torch
from .interpolation_base import InterpolationBase

def jacobian_frobenius_regularization_fn(div_samples, h0, dhdt):
    sqjacnorm = []
    for e in [torch.randn_like(h0) for k in range(div_samples)]:
        e_dhdt_dx = torch.autograd.grad(dhdt, h0, e, create_graph=True)[0]
        # import pdb ; pdb.set_trace()
        n = e_dhdt_dx.flatten(start_dim=1).pow(2).mean(dim=1, keepdim=True)
        sqjacnorm.append(n)
    return torch.cat(sqjacnorm, dim=1).mean(dim=1)

def quadratic_cost( dx):
    dx = dx.view(dx.shape[0], -1)
    return 0.5*dx.pow(2).mean(dim=-1)

def change_by_evolve_way(tensor, evolve_way, state_dim):
    """
    tensor: (batch, state, feature)
    evolve_way -> state_part is changed
    """
    if evolve_way == "concat":
        tensor = tensor.flatten(start_dim = 1)[:,None].repeat(1,state_dim,1)
    elif evolve_way == "sum":
        tensor = tensor.sum(dim=1)[:,None].repeat(1,state_dim,1)
    return tensor        




class VectorField_emb_lstm(torch.nn.Module):
    def __init__(self, func, X_func, prod_method, kinetic, div_samples, residual, evolve_way):
        super().__init__()
        self.func = func
        self.X_func = X_func # derivative -> # (batch, state, num node, feature)
        self.prod_method = prod_method
        self.kinetic = kinetic
        self.div_samples = div_samples
        self.residual = residual
        self.evolve_way = evolve_way


    def forward(self, t, z_feature):
        """
        z_feature[0] (z0)      : (batch, state, node, feature)
        z_feature[1] (z_sq)    : (batch, 1)
        z_feature[2] (z_quad)  : (batch, 1)
        z_feature[3] (e0)      : (batch, state, feature)
        z_feature[4] (e_sq)    : (batch, 1)
        z_feature[5] (e_quad)  : (batch, 1)
        """
        if self.kinetic:
            with torch.set_grad_enabled(True):
                for s_ in z_feature: s_.requires_grad_(True)
                de0_dt_s = self.X_func(t,z_feature[3:])
                for s_ in de0_dt_s: s_.requires_grad_(True)
                de0_dt = de0_dt_s[0]
                dz0_dt = self._forward(t,z_feature[0], z_feature[3], de0_dt)
                for s_ in dz0_dt: s_.requires_grad_(True)
                z1_sqjacnorm = jacobian_frobenius_regularization_fn( self.div_samples, z_feature[0], dz0_dt)
                z1_quad = quadratic_cost(dz0_dt)
            if self.residual:
                out = (dz0_dt - z_feature[0], z1_sqjacnorm, z1_quad) 
            else:
                out = (dz0_dt, z1_sqjacnorm, z1_quad)
        else:
            de0_dt_s = self.X_func(t,z_feature[1:])
            de0_dt = de0_dt_s[0]
            dz0_dt = self._forward(t,z_feature[0], z_feature[1], de0_dt)
            if self.residual:
                out =  (dz0_dt - z_feature[0],)
            else:
                out = (dz0_dt, )
        return out + de0_dt_s


    def _forward(self, t, z0, e0, de_dt):
        if self.prod_method == 'evaluate':
            path_feature = e0 # (batch, state, feature)
            path_feature = change_by_evolve_way(path_feature, self.evolve_way, e0.size(1))
            path_feature = path_feature[:,:,None].repeat(1,1,z0.size(2),1)
            out = self.func(z0, path_feature)

        elif self.prod_method == 'derivative':
            path_feature = de_dt 
            path_feature = change_by_evolve_way(path_feature, self.evolve_way, e0.size(1))
            path_feature = path_feature[:,:,None].repeat(1,1,z0.size(2),1)
            out = self.func(z0, path_feature)
        elif self.prod_method == 'matmul':
            control_gradient = de_dt 
            control_gradient = change_by_evolve_way(control_gradient, self.evolve_way, e0.size(1))
            control_gradient = control_gradient[:,:,None].repeat(1,1,z0.size(2),1) # (batch, # state, # node, feature1)
            vector_field = self.func(z0)     # (batch, # state, # node, feature2, feature1)
            # import pdb ; pdb.set_trace()
            out = torch.einsum("abcj,abcij->abci",control_gradient,vector_field)
        return out
        
class VectorField_emb(torch.nn.Module):
    def __init__(self, func, X, prod_method, kinetic, div_samples, residual, evolve_way):
        super().__init__()
        self.func = func
        self.X = X # derivative -> (#batch, #state, #feature)
        self.prod_method = prod_method
        self.kinetic = kinetic
        self.div_samples = div_samples
        self.residual = residual
        self.evolve_way = evolve_way


    def forward(self, t, z_feature):
        if self.kinetic:
            with torch.set_grad_enabled(True):
                for s_ in z_feature:
                    s_.requires_grad_(True)    
                dz0_dt = self._forward(t,z_feature[0])
                dz0_dt.requires_grad_(True)
                z1_sqjacnorm = jacobian_frobenius_regularization_fn( self.div_samples, z_feature[0], dz0_dt)
                z1_quad = quadratic_cost(dz0_dt)
            if self.residual:
                out = (dz0_dt - z_feature[0], z1_sqjacnorm, z1_quad)
            else:
                out = (dz0_dt, z1_sqjacnorm, z1_quad)
        else:
            z_feature = z_feature[0]
            if self.residual:
                out = (self._forward(t,z_feature) - z_feature,)
            else:
                out = (self._forward(t,z_feature),)
        return out

    def _forward(self, t, z_feature):
        if self.prod_method == 'evaluate':
            path_feature = self.X.evaluate(t) 
            path_feature = change_by_evolve_way(path_feature, self.evolve_way, z_feature.size(1))
            out = self.func(z_feature, path_feature)
        elif self.prod_method == 'derivative':
            path_feature = self.X.derivative(t) #  [X.evaluate(t) for X in self.X]
            path_feature = change_by_evolve_way(path_feature, self.evolve_way, z_feature.size(1))
            out = self.func(z_feature, path_feature)
        elif self.prod_method == 'matmul':
            control_gradient = self.X.derivative(t) # (batch, # state, feature1+alpha)
            vector_field = self.func(z_feature)     # [(batch, feature2, feature1) for i in # state]
            each_dims = [i.size(-1) for i in vector_field]
            control_gradient = change_by_evolve_way(control_gradient, self.evolve_way, z_feature.size(1))
            # import pdb ; pdb.set_trace()
            out = torch.stack([torch.einsum("aj,aij->ai",control_gradient[:,i,:d],vector_field[i]) for i, d in enumerate(each_dims)],dim=1)
            
        return out


def cdeint(e0, emb_func, emb_X, emb_prod_method, emb_kinetic, emb_div_samples, emb_residual, emb_evolve_way,
           z0, lstm_func, lstm_prod_method, lstm_kinetic, lstm_div_samples, lstm_residual, lstm_evolve_way,
           is_coevolving, t, adjoint, **kwargs):
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    h0_name = ["e"]
    h0 = [e0]
    vector_field = VectorField_emb(emb_func, emb_X, emb_prod_method, emb_kinetic, emb_div_samples, emb_residual, emb_evolve_way)
    if emb_kinetic:
        h0 = h0 + [torch.zeros(e0.size(0)).to(e0), torch.zeros(e0.size(0)).to(e0)]
        h0_name = h0_name + ["e_ki_sq", "e_ki_quad"]
    if is_coevolving:
        vector_field = VectorField_emb_lstm(lstm_func, vector_field, lstm_prod_method, lstm_kinetic, lstm_div_samples, lstm_residual, lstm_evolve_way)
        if not lstm_kinetic:
            h0 = [z0] + h0 
            h0_name = ["z"] + h0_name
        else:
            h0 = [z0,torch.zeros(z0.size(0)).to(z0), torch.zeros(z0.size(0)).to(z0)] + h0 
            h0_name = ["z","z_ki_sq", "z_ki_quad"] + h0_name        
    # import pdb ; pdb.set_trace()
    out = odeint(func=vector_field, y0=tuple(h0), t=t, **kwargs)
    return dict(zip(h0_name,out))