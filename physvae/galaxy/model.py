import torch
from torch import nn
from torch.nn import functional as F

from .. import utils
from ..mlp import MLP
from ..unet import UNet
# import sys; sys.path.append('../'); import utils; from mlp import MLP


# NOTE: z_aux(s) is/are z_A in the paper, and z_phy is z_P in the paper


xc = 0.0
yc = 0.0

#                    I0   A      e      theta
feasible_range_lb = [0.1, 0.001, 0.0,   0]
feasible_range_ub = [1.0, 10.0,  0.999, 3.142]


class Decoders(nn.Module):
    def __init__(self, config:dict):
        super(Decoders, self).__init__()

        dim_z_aux2 = config['dim_z_aux2']
        activation = config['activation']
        no_phy = config['no_phy']
        x_lnvar = config['x_lnvar']

        self.register_buffer('param_x_lnvar', torch.ones(1)*x_lnvar)

        if not no_phy:
            if dim_z_aux2 >= 0:
                unet_size = config['unet_size']

                # phy (and aux)
                k = 1
                self.func_aux2_expand1 = MLP([4+dim_z_aux2, 32, 64, 128], activation)
                self.func_aux2_expand2 = nn.Sequential(
                    nn.ConvTranspose2d(128, k*16, 4, 1, 0, bias=False), nn.BatchNorm2d(k*16), nn.ReLU(True),
                    nn.ConvTranspose2d(k*16, k*8, 4, 2, 1, bias=False), nn.BatchNorm2d(k*8), nn.ReLU(True),
                    nn.ConvTranspose2d(k*8, k*4, 4, 2, 1, bias=False), nn.BatchNorm2d(k*4), nn.ReLU(True),
                    nn.ConvTranspose2d(k*4, k*2, 5, 2, 1, bias=False), nn.BatchNorm2d(k*2), nn.ReLU(True),
                    nn.ConvTranspose2d(k*2, 1, 5, 2, 0, bias=False),
                )
                k = 4
                self.func_aux2_map = UNet(unet_size)
        else:
            # no phy
            k = 4
            self.func_aux2 = nn.Sequential(
                nn.ConvTranspose2d(dim_z_aux2, k*32, 4, 1, 0, bias=False), nn.BatchNorm2d(k*32), nn.ReLU(True),
                nn.ConvTranspose2d(k*32, k*16, 4, 2, 1, bias=False), nn.BatchNorm2d(k*16), nn.ReLU(True),
                nn.ConvTranspose2d(k*16, k*8, 4, 2, 1, bias=False), nn.BatchNorm2d(k*8), nn.ReLU(True),
                nn.ConvTranspose2d(k*8, k*4, 5, 2, 1, bias=False), nn.BatchNorm2d(k*4), nn.ReLU(True),
                nn.ConvTranspose2d(k*4, 3, 5, 2, 0, bias=False),
            )


class Encoders(nn.Module):
    def __init__(self, config:dict):
        super(Encoders, self).__init__()

        dim_z_aux2 = config['dim_z_aux2']
        activation = config['activation']
        no_phy = config['no_phy']
        num_units_feat = config['num_units_feat']

        self.func_feat = FeatureExtractor(config)

        if dim_z_aux2 > 0:
            hidlayers_aux2_enc = config['hidlayers_aux2_enc']
            self.func_z_aux2_mean = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)
            self.func_z_aux2_lnvar = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)

        if not no_phy:
            hidlayers_z_phy = config['hidlayers_z_phy']
            self.func_unmixer_coeff = nn.Sequential(MLP([num_units_feat, 16, 16, 3], activation), nn.Tanh())
            self.func_z_phy_mean = nn.Sequential(MLP([num_units_feat,]+hidlayers_z_phy+[4,], activation), nn.Softplus())
            self.func_z_phy_lnvar = MLP([num_units_feat,]+hidlayers_z_phy+[4,], activation)


class FeatureExtractor(nn.Module):
    def __init__(self, config:dict, in_channels:int=3):
        super(FeatureExtractor, self).__init__()

        self.in_channels = in_channels
        activation = config['activation']
        num_units_feat = config['num_units_feat']
        k = 8
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, k*2, 5, 1, 2), nn.BatchNorm2d(k*2), nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2), # 2k x34x34
            nn.Conv2d(k*2, k*4, 5, 1, 2), nn.BatchNorm2d(k*4), nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2), # 4k x17x17
            nn.Conv2d(k*4, k*8, 5, 1, 2), nn.BatchNorm2d(k*8), nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2), # 8k x8x8
            nn.Conv2d(k*8, k*16, 5, 1, 2), nn.BatchNorm2d(k*16), nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2), # 16k x4x4
        )
        self.after = nn.Linear(k*16*16, num_units_feat, bias=False)

    def forward(self, x:torch.Tensor):
        x_ = x.view(-1, self.in_channels, 69, 69)
        n = x_.shape[0]
        return self.after(self.convnet(x_).view(n,-1))


class Physics(nn.Module):
    def __init__(self):
        super(Physics, self).__init__()
        self.register_buffer('x_coord', torch.linspace(-1.0, 1.0, 69).unsqueeze(0).repeat(69,1))
        self.register_buffer('y_coord', torch.flipud(torch.linspace(-1.0, 1.0, 69).unsqueeze(1).repeat(1,69)))

    def forward(self, z_phy:torch.Tensor):
        # z_phy = [I0, A, e, theta]
        n = z_phy.shape[0]

        I0 = z_phy[:, 0].view(n,1,1)
        A = z_phy[:, 1].view(n,1,1)
        e = z_phy[:, 2].view(n,1,1)
        B = A * (1.0 - e)
        theta = z_phy[:, 3].view(n,1,1)

        x = self.x_coord.unsqueeze(0).expand(n,69,69)
        y = self.y_coord.unsqueeze(0).expand(n,69,69)
        x_rotated = torch.cos(theta)*(x-xc) - torch.sin(theta)*(y-yc)
        y_rotated = torch.sin(theta)*(x-xc) + torch.cos(theta)*(y-yc)
        xx = x_rotated / A
        yy = y_rotated / B
        # r = torch.sqrt((x_rotated/A).pow(2) + (y_rotated/B).pow(2))
        r = torch.norm(torch.cat([xx.unsqueeze(0), yy.unsqueeze(0)], dim=0), dim=0) # (n,69,69)

        out = I0 * torch.exp(-r)
        return out.unsqueeze(1) # (n,1,69,69) is returned; not (n,3,69,69)


class VAE(nn.Module):
    def __init__(self, config:dict):
        super(VAE, self).__init__()

        assert config['range_I0'][0] <= config['range_I0'][1]
        assert config['range_A'][0] <= config['range_A'][1]
        assert config['range_e'][0] <= config['range_e'][1]
        assert config['range_theta'][0] <= config['range_theta'][1]

        self.dim_z_aux2 = config['dim_z_aux2']
        self.activation = config['activation']
        self.no_phy = config['no_phy']

        self.range_I0 = config['range_I0']
        self.range_A = config['range_A']
        self.range_e = config['range_e']
        self.range_theta = config['range_theta']

        # Decoding part
        self.dec = Decoders(config)

        # Encoding part
        self.enc = Encoders(config)

        # Physics
        self.physics_model = Physics()

        self.register_buffer('feasible_range_lb', torch.Tensor(feasible_range_lb))
        self.register_buffer('feasible_range_ub', torch.Tensor(feasible_range_ub))


    def priors(self, n:int, device:torch.device):
        prior_z_phy_mean = torch.cat([
            torch.ones(n,1,device=device) * 0.5 * (self.range_I0[0] + self.range_I0[1]),
            torch.ones(n,1,device=device) * 0.5 * (self.range_A[0] + self.range_A[1]),
            torch.ones(n,1,device=device) * 0.5 * (self.range_e[0] + self.range_e[1]),
            torch.ones(n,1,device=device) * 0.5 * (self.range_theta[0] + self.range_theta[1]),
        ], dim=1)
        prior_z_phy_std = torch.cat([
            torch.ones(n,1,device=device) * max(1e-3, 0.866*(self.range_I0[1] - self.range_I0[0])),
            torch.ones(n,1,device=device) * max(1e-3, 0.866*(self.range_A[1] - self.range_A[0])),
            torch.ones(n,1,device=device) * max(1e-3, 0.866*(self.range_e[1] - self.range_e[0])),
            torch.ones(n,1,device=device) * max(1e-3, 0.866*(self.range_theta[1] - self.range_theta[0])),
        ], dim=1)
        prior_z_phy_stat = {'mean': prior_z_phy_mean, 'lnvar': 2.0*torch.log(prior_z_phy_std)}
        prior_z_aux2_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux2), device=device),
            'lnvar': torch.zeros(n, max(0,self.dim_z_aux2), device=device)}
        return prior_z_phy_stat, prior_z_aux2_stat


    def generate_physonly(self, z_phy:torch.Tensor):
        y = self.physics_model(z_phy) # (n,1,69,69)
        return y


    def decode(self, z_phy:torch.Tensor, z_aux2:torch.Tensor, full:bool=False):
        if not self.no_phy:
            # with physics
            y = self.physics_model(z_phy) # (n,1,69,69)
            x_P = y.repeat(1,3,1,1)
            if self.dim_z_aux2 >= 0:
                expanded1 = self.dec.func_aux2_expand1(torch.cat([z_phy, z_aux2], dim=1))
                expanded2 = self.dec.func_aux2_expand2(expanded1.view(-1,128,1,1))
                x_PB = torch.sigmoid(self.dec.func_aux2_map(torch.cat([x_P, expanded2], dim=1)))
            else:
                x_PB = x_P.clone()
        else:
            # no physics
            y = torch.zeros(z_phy.shape[0], 1, 69, 69)
            if self.dim_z_aux2 >= 0:
                x_PB = torch.sigmoid(self.dec.func_aux2(z_aux2.view(-1, self.dim_z_aux2, 1, 1)))
            else:
                x_PB = torch.zeros(z_phy.shape[0], 3, 69, 69)
            x_P = x_PB.clone()

        if full:
            return x_PB, x_P, self.dec.param_x_lnvar, y
        else:
            return x_PB, self.dec.param_x_lnvar


    def encode(self, x:torch.Tensor):
        x_ = x.view(-1, 3, 69, 69)
        n = x_.shape[0]
        device = x_.device

        # infer z_aux2
        feature_aux2 = self.enc.func_feat(x_)
        if self.dim_z_aux2 > 0:
            z_aux2_stat = {'mean':self.enc.func_z_aux2_mean(feature_aux2), 'lnvar':self.enc.func_z_aux2_lnvar(feature_aux2)}
        else:
            z_aux2_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        # infer z_phy
        if not self.no_phy:
            coeff = self.enc.func_unmixer_coeff(feature_aux2) # (n,3)
            unmixed = torch.sum(x_*coeff.unsqueeze(2).unsqueeze(3), dim=1, keepdim=True)
            feature_phy = self.enc.func_feat(unmixed.expand(n, 3, 69, 69))
            z_phy_stat = {'mean': self.enc.func_z_phy_mean(feature_phy), 'lnvar': self.enc.func_z_phy_lnvar(feature_phy)}
        else:
            unmixed = torch.zeros(n, 3, 0, 0, device=device)
            z_phy_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        return z_phy_stat, z_aux2_stat, unmixed


    def draw(self, z_phy_stat:dict, z_aux2_stat:dict, hard_z:bool=False):
        if not hard_z:
            z_phy = utils.draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])
            z_aux2 = utils.draw_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'])
        else:
            z_phy = z_phy_stat['mean'].clone()
            z_aux2 = z_aux2_stat['mean'].clone()

        # cut infeasible regions
        if not self.no_phy:
            n = z_phy.shape[0]
            z_phy = torch.max(self.feasible_range_lb.unsqueeze(0).expand(n,4), z_phy)
            z_phy = torch.min(self.feasible_range_ub.unsqueeze(0).expand(n,4), z_phy)

        return z_phy, z_aux2


    def forward(self, x:torch.Tensor, reconstruct:bool=True, hard_z:bool=False):
        z_phy_stat, z_aux2_stat, unmixed, = self.encode(x)

        if not reconstruct:
            return z_phy_stat, z_aux2_stat

        # draw & reconstruction
        x_mean, x_lnvar = self.decode(*self.draw(z_phy_stat, z_aux2_stat, hard_z=hard_z), full=False)

        return z_phy_stat, z_aux2_stat, x_mean, x_lnvar
