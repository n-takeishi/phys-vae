""" ODE-based physics-augmented VAE model.
"""

import copy
import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint

from .. import utils
from ..mlp import MLP
# import sys; sys.path.append('../'); import utils; from mlp import MLP


# NOTE: z_aux(s) is/are z_A in the paper, and z_phy is z_P in the paper


dcoeff_feasible_range = [5e-3, 2e-1]


class Decoders(nn.Module):
    def __init__(self, config:dict):
        super(Decoders, self).__init__()

        dim_x = config['dim_x']
        dim_y = dim_x
        dim_t = config['dim_t']
        dim_z_aux1 = config['dim_z_aux1']
        dim_z_aux2 = config['dim_z_aux2']
        activation = config['activation']
        no_phy = config['no_phy']
        x_lnvar = config['x_lnvar']

        # x_lnvar
        self.register_buffer('param_x_lnvar', torch.ones(1)*x_lnvar)

        if dim_z_aux1 >= 0:
            hidlayers_aux1 = config['hidlayers_aux1_dec']

            # z_aux1, y, sin(t), cos(t) --> time-derivative of y
            self.func_aux1 = MLP([dim_z_aux1+dim_y+2,]+hidlayers_aux1+[dim_y,], activation)

        self.func_aux2_map = nn.Linear(dim_y, dim_x)
        if dim_z_aux2 >= 0:
            hidlayers_aux2 = config['hidlayers_aux2_dec']

            # y_seq, z_aux2 --> x
            dim_z_phy = 0 if no_phy else 1
            self.func_aux2_res = MLP([dim_z_phy+max(0,dim_z_aux1)+dim_z_aux2,]+hidlayers_aux2+[dim_x*dim_t,], activation)


class Encoders(nn.Module):
    def __init__(self, config:dict):
        super(Encoders, self).__init__()

        dim_x = config['dim_x']
        dim_y = dim_x
        dim_t = config['dim_t']
        dim_z_aux1 = config['dim_z_aux1']
        dim_z_aux2 = config['dim_z_aux2']
        activation = config['activation']
        no_phy = config['no_phy']
        num_units_feat = config['num_units_feat']

        if dim_z_aux1 > 0:
            hidlayers_aux1_enc = config['hidlayers_aux1_enc']

            # x --> feature_aux1
            self.func_feat_aux1 = FeatureExtractor(config)

            # feature_aux1 --> z_aux1
            self.func_z_aux1_mean = MLP([num_units_feat,]+hidlayers_aux1_enc+[dim_z_aux1,], activation)
            self.func_z_aux1_lnvar = MLP([num_units_feat,]+hidlayers_aux1_enc+[dim_z_aux1,], activation)

        if dim_z_aux2 > 0:
            hidlayers_aux2_enc = config['hidlayers_aux2_enc']

            # x --> feature_aux2
            self.func_feat_aux2 = FeatureExtractor(config)

            # feature_aux2 --> z_aux2
            self.func_z_aux2_mean = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)
            self.func_z_aux2_lnvar = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)

        if not no_phy:
            hidlayers_unmixer = config['hidlayers_unmixer']
            hidlayers_dcoeff = config['hidlayers_dcoeff']

            # x, z_aux1, z_aux2 --> unmixed (~= y)
            self.func_unmixer_map = nn.Linear(dim_x, dim_y)
            if dim_z_aux1>0 or dim_z_aux2>0:
                self.func_unmixer_res = MLP([max(dim_z_aux1,0)+max(dim_z_aux2,0),]+hidlayers_unmixer+[dim_y*dim_t,], activation)
                # self.func_unmixer_rnn = nn.GRU(dim_x, dim_y, num_layers=2, bidirectional=False)
                # self.func_unmixer = MLP([dim_x*dim_t+max(dim_z_aux1,0)+max(dim_z_aux2,0),]+[256,256,256]+[dim_y*dim_t,], activation)

            # unmixed --> feature_phy
            config_ = copy.deepcopy(config); config_['dim_x'] = dim_y
            self.func_feat_phy = FeatureExtractor(config_)

            # features_phy --> dcoeff
            self.func_dcoeff_mean = nn.Sequential(MLP([num_units_feat,]+hidlayers_dcoeff+[1,], activation), nn.Softplus())
            self.func_dcoeff_lnvar = MLP([num_units_feat,]+hidlayers_dcoeff+[1,], activation)


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
    def __init__(self, dx:float):
        super(Physics, self).__init__()
        self.dx = dx
        self.register_buffer('discLap2', torch.Tensor([1.0, -2.0, 1.0]).unsqueeze(0).unsqueeze(0))
        self.discLap2.requires_grad=False

    def forward(self, dcoeff:torch.Tensor, y:torch.Tensor):
        """
        given parameter and y, return dy/dt
        [state]
            y: shape <n x dim_y>, dim_y=dim_x basically
        [physics parameter]
            dcoeff: shape <n x 1>
        """
        y_xx = F.conv1d(y.unsqueeze(1), self.discLap2, bias=None, stride=1, padding=1).squeeze(1)
        return dcoeff*y_xx/self.dx/ self.dx


class VAE(nn.Module):
    def __init__(self, config:dict):
        super(VAE, self).__init__()

        assert config['range_dcoeff'][0] <= config['range_dcoeff'][1]

        self.dim_x = config['dim_x']
        self.dim_y = self.dim_x
        self.dim_t = config['dim_t']
        self.dim_z_aux1 = config['dim_z_aux1']
        self.dim_z_aux2 = config['dim_z_aux2']
        self.range_dcoeff = config['range_dcoeff']
        self.activation = config['activation']
        self.dx = config['dx']
        self.dt = config['dt']
        self.intg_lev = config['intg_lev']
        self.ode_solver = config['ode_solver']
        self.no_phy = config['no_phy']

        # Decoding part
        self.dec = Decoders(config)

        # Encoding part
        self.enc = Encoders(config)

        # Physics
        self.physics_model = Physics(self.dx)

        # set time indices for integration
        self.dt_intg = self.dt / float(self.intg_lev)
        self.len_intg = (self.dim_t - 1) * self.intg_lev + 1
        self.register_buffer('t_intg', torch.linspace(0.0, self.dt_intg*(self.len_intg-1), self.len_intg))


    def priors(self, n:int, device:torch.device):
        prior_dcoeff_stat = {'mean': torch.ones(n,1,device=device) * 0.5 * (self.range_dcoeff[0] + self.range_dcoeff[1]),
            'lnvar': 2.0*torch.log( torch.ones(n,1,device=device) * max(1e-3, 0.866*(self.range_dcoeff[1] - self.range_dcoeff[0])) )}
        prior_z_aux1_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux1), device=device),
            'lnvar': torch.zeros(n, max(0,self.dim_z_aux1), device=device)}
        prior_z_aux2_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux2), device=device),
            'lnvar': torch.zeros(n, max(0,self.dim_z_aux2), device=device)}
        return prior_dcoeff_stat, prior_z_aux1_stat, prior_z_aux2_stat


    def generate_physonly(self, dcoeff:torch.Tensor, init_y:torch.Tensor):
        n = dcoeff.shape[0]
        device = dcoeff.device

        # define ODE
        def ODEfunc(t:torch.Tensor, y:torch.Tensor):
            return self.physics_model(dcoeff, y)

        # solve ODE
        y_seq = odeint(ODEfunc, init_y, self.t_intg, method=self.ode_solver) # <len_intg x n x dim_y>
        y_seq = y_seq[range(0, self.len_intg, self.intg_lev)].permute(1,2,0).contiguous() # subsample and reshape to <n x dim_y x dim_t>

        return y_seq


    def decode(self, dcoeff:torch.Tensor, z_aux1:torch.Tensor, z_aux2:torch.Tensor, init_y:torch.Tensor, full:bool=False):
        n = dcoeff.shape[0]
        device = dcoeff.device

        # define ODE
        def ODEfunc(t:torch.Tensor, _y:torch.Tensor):
            """
            - t: scalar
            - _y: shape (n, 2dim_y) or (n, dim_y)
            """

            y_PA = _y[:,0:self.dim_y]
            if full:
                y_P = _y[:,self.dim_y:]

            if not self.no_phy:
                # physics part (dcoeff & y --> time-deriv of y)
                y_dot_phy_PA = self.physics_model(dcoeff, y_PA)
                if full:
                    y_dot_phy_P = self.physics_model(dcoeff, y_P)
            else:
                # when model has no physics part *originally*
                y_dot_phy_PA = torch.zeros(n, self.dim_y, device=device)
                if full:
                    y_dot_phy_P = torch.zeros(n, self.dim_y, device=device)

            if self.dim_z_aux1 >= 0:
                # in-ODE auxiliary part (z_aux1, y, sin(t), cos(t) --> time-deriv of y)
                tmp = torch.cat([z_aux1, y_PA, torch.sin(t).expand(n,1), torch.cos(t).expand(n,1)], dim=1)
                y_dot_aux_PA =  self.dec.func_aux1(tmp)
            else:
                # when model has no in-ODE auxiliary part *originally*
                y_dot_aux_PA = torch.zeros(n, self.dim_y, device=device)

            if full:
                return torch.cat([y_dot_phy_PA+y_dot_aux_PA, y_dot_phy_P], dim=1)
            else:
                return torch.cat([y_dot_phy_PA+y_dot_aux_PA], dim=1)

        # solve
        if full:
            initcond = torch.cat([init_y, init_y], dim=1) # <n x 2dim_y>
        else:
            initcond = init_y.clone() # <n x dim_y>
        y_seq = odeint(ODEfunc, initcond, self.t_intg, method=self.ode_solver) # <len_intg x n x 2dim_y or dim_y>
        y_seq = y_seq[range(0, self.len_intg, self.intg_lev)] # subsample to <dim_t x n x 2dim_y or dim_y>

        # extract to <n x dim_y x dim_t>
        y_seq_PA = y_seq[:,:,0:self.dim_y].permute(1,2,0).contiguous()
        if full:
            y_seq_P = y_seq[:,:,self.dim_y:].permute(1,2,0).contiguous()

        # out-ODE auxiliary part (y_seq, z_aux2 --> x)
        x_PA = self.dec.func_aux2_map(y_seq_PA.permute(0,2,1)).permute(0,2,1); x_PAB = x_PA.clone()
        if full:
            x_P = self.dec.func_aux2_map(y_seq_P.permute(0,2,1)).permute(0,2,1); x_PB = x_P.clone()
        if self.dim_z_aux2 >= 0:
            x_PAB += self.dec.func_aux2_res(torch.cat([dcoeff, z_aux1, z_aux2], dim=1)).reshape(-1, self.dim_x, self.dim_t)
            if full:
                x_PB += self.dec.func_aux2_res(torch.cat([dcoeff, z_aux1, z_aux2], dim=1)).reshape(-1, self.dim_x, self.dim_t)

        if full:
            return x_PAB, x_PA, x_PB, x_P, self.dec.param_x_lnvar
        else:
            return x_PAB, self.dec.param_x_lnvar


    def encode(self, x:torch.Tensor):
        x_ = x.view(-1, self.dim_x, self.dim_t)
        n = x_.shape[0]
        device = x_.device

        # infer z_aux1, z_aux2
        if self.dim_z_aux1 > 0:
            feature_aux1 = self.enc.func_feat_aux1(x_)
            z_aux1_stat = {'mean':self.enc.func_z_aux1_mean(feature_aux1), 'lnvar':self.enc.func_z_aux1_lnvar(feature_aux1)}
        else:
            z_aux1_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        if self.dim_z_aux2 > 0:
            feature_aux2 = self.enc.func_feat_aux2(x_)
            z_aux2_stat = {'mean':self.enc.func_z_aux2_mean(feature_aux2), 'lnvar':self.enc.func_z_aux2_lnvar(feature_aux2)}
        else:
            z_aux2_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        # infer dcoeff
        if not self.no_phy:
            unmixed = self.enc.func_unmixer_map(x_.permute(0,2,1)).permute(0,2,1).contiguous()
            if self.dim_z_aux1>0 or self.dim_z_aux2>0:
                unmixed += self.enc.func_unmixer_res(torch.cat([z_aux1_stat['mean'], z_aux2_stat['mean']], dim=1)).view(-1, self.dim_y, self.dim_t)

            # unmixed, _ = self.enc.func_unmixer_rnn(x_.permute(2,0,1), torch.zeros(2, n, self.dim_y, device=device))
            # unmixed = unmixed.permute(1,2,0).contiguous()

            # unmixed = self.enc.func_unmixer(torch.cat([x_.view(n,-1), z_aux1_stat['mean'], z_aux2_stat['mean']], dim=1)).view(-1, self.dim_y, self.dim_t)

            feature_phy = self.enc.func_feat_phy(unmixed)
            dcoeff_stat = {'mean': self.enc.func_dcoeff_mean(feature_phy), 'lnvar': self.enc.func_dcoeff_lnvar(feature_phy)}
        else:
            unmixed = torch.zeros(n, self.dim_y, self.dim_t, device=device)
            dcoeff_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        return dcoeff_stat, z_aux1_stat, z_aux2_stat, unmixed


    def draw(self, dcoeff_stat:dict, z_aux1_stat:dict, z_aux2_stat:dict, hard_z:bool=False):
        if not hard_z:
            dcoeff = utils.draw_normal(dcoeff_stat['mean'], dcoeff_stat['lnvar'])
            z_aux1 = utils.draw_normal(z_aux1_stat['mean'], z_aux1_stat['lnvar'])
            z_aux2 = utils.draw_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'])
        else:
            dcoeff = dcoeff_stat['mean'].clone()
            z_aux1 = z_aux1_stat['mean'].clone()
            z_aux2 = z_aux2_stat['mean'].clone()

        # cut infeasible regions
        dcoeff = torch.max(torch.ones_like(dcoeff)*dcoeff_feasible_range[0], dcoeff)
        dcoeff = torch.min(torch.ones_like(dcoeff)*dcoeff_feasible_range[1], dcoeff)

        return dcoeff, z_aux1, z_aux2


    def forward(self, x:torch.Tensor, reconstruct:bool=True, hard_z:bool=False):
        dcoeff_stat, z_aux1_stat, z_aux2_stat, unmixed = self.encode(x)

        if not reconstruct:
            return dcoeff_stat, z_aux1_stat, z_aux2_stat

        # draw & reconstruction
        init_y = x[:,:,0].clone()
        x_mean, x_lnvar = self.decode(*self.draw(dcoeff_stat, z_aux1_stat, z_aux2_stat, hard_z=hard_z), init_y, full=False)

        return dcoeff_stat, z_aux1_stat, z_aux2_stat, x_mean, x_lnvar
