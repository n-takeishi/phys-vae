""" Hamiltonian ODE-based physics-augmented VAE model.
"""

import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad
from torchdiffeq import odeint

from .. import utils
from ..mlp import MLP
# import sys; sys.path.append('../'); import utils; from mlp import MLP


# NOTE: z_aux(s) is/are z_A in the paper, and z_phy is z_P in the paper


class Decoders(nn.Module):
    def __init__(self, config:dict):
        super(Decoders, self).__init__()

        dim_y = config['dim_y']
        dim_x = config['dim_x']
        dim_t = config['dim_t']
        dim_z_phy = config['dim_z_phy']
        dim_z_aux2 = config['dim_z_aux2']
        activation = config['activation']
        no_phy = config['no_phy']
        x_lnvar = config['x_lnvar']

        # x_lnvar
        self.register_buffer('param_x_lnvar', torch.ones(1)*x_lnvar)

        self.func_aux2_map = nn.Linear(dim_y, dim_x)
        if dim_z_aux2 >= 0:
            hidlayers_aux2 = config['hidlayers_aux2_dec']
            self.func_aux2_res = MLP([dim_z_aux2,]+hidlayers_aux2+[dim_x*dim_t,], activation)


class Encoders(nn.Module):
    def __init__(self, config:dict):
        super(Encoders, self).__init__()

        dim_y = config['dim_y']
        dim_x = config['dim_x']
        dim_t = config['dim_t']
        dim_z_phy = config['dim_z_phy']
        dim_z_aux2 = config['dim_z_aux2']
        activation = config['activation']
        no_phy = config['no_phy']
        num_units_feat = config['num_units_feat']
        hidlayers_init_yy = config['hidlayers_init_yy']

        # x --> feature
        self.func_feat = FeatureExtractor(config)

        if dim_z_aux2 > 0:
            hidlayers_aux2_enc = config['hidlayers_aux2_enc']

            # feature --> z_aux2
            self.func_z_aux2_mean = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)
            self.func_z_aux2_lnvar = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)

        if not no_phy and dim_z_phy > 0:
            hidlayers_unmixer = config['hidlayers_unmixer']
            hidlayers_z_phy = config['hidlayers_z_phy']

            # features --> z_phy
            self.func_z_phy_mean = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation)
            self.func_z_phy_lnvar = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation)

        # feature --> init_yy
        self.func_init_yy = MLP([num_units_feat,]+hidlayers_init_yy+[2*dim_y,], activation)


class FeatureExtractor(nn.Module):
    def __init__(self, config:dict):
        super(FeatureExtractor, self).__init__()

        dim_x = config['dim_x']
        dim_t = config['dim_t']
        activation = config['activation']
        arch_feat = config['arch_feat']
        num_units_feat = config['num_units_feat']

        self.dim_x = dim_x
        self.dim_t = dim_t
        self.arch_feat = arch_feat
        self.num_units_feat = num_units_feat

        if arch_feat=='mlp':
            hidlayers_feat = config['hidlayers_feat']

            self.func= MLP([dim_x*dim_t,]+hidlayers_feat+[num_units_feat,], activation, actfun_output=True)
        elif arch_feat=='rnn':
            num_rnns_feat = config['num_rnns_feat']

            self.num_rnns_feat = num_rnns_feat
            self.func = nn.GRU(dim_x, num_units_feat, num_layers=num_rnns_feat, bidirectional=False)
        else:
            raise ValueError('unknown feature type')

    def forward(self, x:torch.Tensor):
        x_ = x.view(-1, self.dim_x, self.dim_t)
        n = x_.shape[0]
        device = x_.device

        if self.arch_feat=='mlp':
            feat = self.func(x_.view(n,-1))
        elif self.arch_feat=='rnn':
            h_0 = torch.zeros(self.num_rnns_feat, n, self.num_units_feat, device=device)
            out, h_n = self.func(x_.permute(2, 0, 1), h_0)
            feat = out[-1]

        return feat


class Physics(nn.Module):
    def __init__(self, config:dict):
        super(Physics, self).__init__()

        activation = config['activation']
        dim_y = config['dim_y']
        dim_z_phy = config['dim_z_phy']
        hidlayers_H = config['hidlayers_H']

        self.dim_y = dim_y
        self.H = MLP([2*dim_y+dim_z_phy,]+hidlayers_H+[1,], activation)

    def forward(self, z_phy:torch.Tensor, yy:torch.Tensor):
        """
        given parameter and yy, return dyy/dt
        [state]
            yy: shape <n x 2dim_y>; the first half should be q (generalized position), the latter half should be p (generalized momentum)
        [physics parameter]
            z_phy: shape <n x dim_z_phy>
        """
        # yy = [q, p]
        H_val = self.H(torch.cat([yy, z_phy], dim=1))
        H_grad = grad([h for h in H_val], [yy], create_graph=self.training, only_inputs=True)[0]
        dHdq = H_grad[:, 0:self.dim_y]
        dHdp = H_grad[:, self.dim_y:]
        return torch.cat([dHdp, -dHdq], dim=1)


class VAE(nn.Module):
    def __init__(self, config:dict):
        super(VAE, self).__init__()

        self.dim_y = config['dim_y']
        self.dim_x = config['dim_x']
        self.dim_t = config['dim_t']
        self.dim_z_phy = config['dim_z_phy']
        self.dim_z_aux2 = config['dim_z_aux2']
        self.activation = config['activation']
        self.dt = config['dt']
        self.intg_lev = config['intg_lev']
        self.ode_solver = config['ode_solver']
        self.no_phy = config['no_phy']

        # Decoding part
        self.dec = Decoders(config)

        # Encoding part
        self.enc = Encoders(config)

        # Physics
        self.physics_model = Physics(config)

        # set time indices for integration
        self.dt_intg = self.dt / float(self.intg_lev)
        self.len_intg = (self.dim_t - 1) * self.intg_lev + 1
        self.register_buffer('t_intg', torch.linspace(0.0, self.dt_intg*(self.len_intg-1), self.len_intg))


    def priors(self, n:int, device:torch.device):
        prior_z_phy_stat = {'mean': torch.zeros(n, max(0,self.dim_z_phy), device=device),
            'lnvar': torch.zeros(n, max(0,self.dim_z_phy), device=device)}
        prior_z_aux2_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux2), device=device),
            'lnvar': torch.zeros(n, max(0,self.dim_z_aux2), device=device)}
        return prior_z_phy_stat, prior_z_aux2_stat


    def generate_physonly(self, z_phy:torch.Tensor, init_yy:torch.Tensor):
        n = z_phy.shape[0]
        device = z_phy.device

        # define ODE
        def ODEfunc(t:torch.Tensor, yy:torch.Tensor):
            return self.physics_model(z_phy, yy)

        # solve ODE
        yy_seq = odeint(ODEfunc, init_yy, self.t_intg, method=self.ode_solver) # <len_intg x n x 2dim_y>
        y_seq = yy_seq[range(0, self.len_intg, self.intg_lev), :, 0:self.dim_y].permute(1,2,0).contiguous() # subsample, extract and reshape to <n x dim_y x dim_t>

        return y_seq


    def decode(self, z_phy:torch.Tensor, z_aux2:torch.Tensor, init_yy:torch.Tensor, full:bool=False):
        n = z_phy.shape[0]
        device = z_phy.device

        # physics part
        if not self.no_phy:
            y_seq_P = self.generate_physonly(z_phy, init_yy)
        else:
            y_seq_P = init_yy[:, 0:self.dim_y].unsqueeze(2).repeat(1, 1, self.dim_t) # (n, dim_y, dim_t)

        x_P = self.dec.func_aux2_map(y_seq_P.permute(0,2,1)).permute(0,2,1).contiguous()

        # out-ODE auxiliary part (y_seq, z_aux2 --> x)
        if self.dim_z_aux2 >= 0:
            x_PB = x_P + self.dec.func_aux2_res(z_aux2).reshape(-1, self.dim_x, self.dim_t)
        else:
            x_PB = x_P.clone()

        if full:
            return x_PB, x_P, self.dec.param_x_lnvar, y_seq_P
        else:
            return x_PB, self.dec.param_x_lnvar


    def encode(self, x:torch.Tensor):
        x_ = x.view(-1, self.dim_x, self.dim_t)
        n = x_.shape[0]
        device = x_.device

        feature = self.enc.func_feat(x_)

        # infer z_aux2
        if self.dim_z_aux2 > 0:
            z_aux2_stat = {'mean':self.enc.func_z_aux2_mean(feature), 'lnvar':self.enc.func_z_aux2_lnvar(feature)}
        else:
            z_aux2_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        # infer z_phy
        if not self.no_phy and self.dim_z_phy > 0:
            z_phy_stat = {'mean': self.enc.func_z_phy_mean(feature), 'lnvar': self.enc.func_z_phy_lnvar(feature)}
        else:
            z_phy_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        init_yy = self.enc.func_init_yy(feature)

        return z_phy_stat, z_aux2_stat, init_yy


    def draw(self, z_phy_stat:dict, z_aux2_stat:dict, hard_z:bool=False):
        if not hard_z:
            z_phy = utils.draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])
            z_aux2 = utils.draw_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'])
        else:
            z_phy = z_phy_stat['mean'].clone()
            z_aux2 = z_aux2_stat['mean'].clone()

        return z_phy, z_aux2


    def forward(self, x:torch.Tensor, reconstruct:bool=True, hard_z:bool=False):
        z_phy_stat, z_aux2_stat, init_yy = self.encode(x)

        if not reconstruct:
            return z_phy_stat, z_aux2_stat

        # draw & reconstruction
        x_mean, x_lnvar = self.decode(*self.draw(z_phy_stat, z_aux2_stat, hard_z=hard_z), init_yy, full=False)

        return z_phy_stat, z_aux2_stat, x_mean, x_lnvar
