import argparse
import os
import json
import time
import numpy as np

import torch
from torch import optim
import torch.utils.data
from torchvision import transforms

from .model import VAE
from .. import utils
# from model import VAE
# import sys; sys.path.append('../'); import utils


def rescale(*args):
    return [torch.sqrt(x+1e-3) for x in args]


def set_parser():
    parser = argparse.ArgumentParser(description='')

    # input/output setting
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)

    # prior knowledge
    parser.add_argument('--range-I0', type=float, nargs='+', default=[0.1, 1.0])
    parser.add_argument('--range-A', type=float, nargs='+', default=[0.1, 1.5])
    parser.add_argument('--range-e', type=float, nargs='+', default=[0.0, 0.999])
    parser.add_argument('--range-theta', type=float, nargs='+', default=[0.0, 3.142])

    # model (general)
    parser.add_argument('--dim-z-aux2', type=int, required=True, help="if 0, aux2 is still alive without latent variable; set -1 to deactivate")
    parser.add_argument('--activation', type=str, default='elu') #choices=['relu','leakyrelu','elu','softplus','prelu'],
    parser.add_argument('--no-phy', action='store_true', default=False)

    # model (decoder)
    parser.add_argument('--unet-size', type=int, default=2)
    parser.add_argument('--x-lnvar', type=float, default=-8.0)

    # model (encoder)
    parser.add_argument('--hidlayers-aux2-enc', type=int, nargs='+', default=[128,])
    parser.add_argument('--hidlayers-z-phy', type=int, nargs='+', default=[128,])
    parser.add_argument('--num-units-feat', type=int, default=512)

    # optimization (base)
    parser.add_argument('--data-augmentation', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--adam-eps', type=float, default=1e-3)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--epochs-pretrain', type=int, default=50)
    parser.add_argument('--balance-kld', type=float, default=1.0)
    parser.add_argument('--balance-unmix', type=float, default=0.0)
    parser.add_argument('--balance-dataug', type=float, default=0.0)
    parser.add_argument('--balance-lact-dec', type=float, default=0.0)
    parser.add_argument('--balance-lact-enc', type=float, default=0.0)
    parser.add_argument('--rescale', action='store_true', default=False)

    # data usage
    parser.add_argument('--train-size', type=int, default=400)
    parser.add_argument('--valid-size', type=int, default=50)

    # others
    parser.add_argument('--save-interval', type=int, default=999999999)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1234567890)

    return parser


def loss_function(args, data, z_phy_stat, z_aux2_stat, x):
    n = data.shape[0]
    device = data.device

    recerr_sq = torch.sum((x - data).pow(2), dim=[1,2,3]).mean()

    prior_z_phy_stat, prior_z_aux2_stat = model.priors(n, device)

    KL_z_aux2 = utils.kldiv_normal_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'],
        prior_z_aux2_stat['mean'], prior_z_aux2_stat['lnvar']) if args.dim_z_aux2 > 0 else torch.zeros(1, device=device)
    KL_z_phy = utils.kldiv_normal_normal(z_phy_stat['mean'], z_phy_stat['lnvar'],
        prior_z_phy_stat['mean'], prior_z_phy_stat['lnvar']) if not args.no_phy else torch.zeros(1, device=device)

    kldiv = (KL_z_aux2 + KL_z_phy).mean()

    return recerr_sq, kldiv


def train(epoch, args, device, loader, model, optimizer):
    model.train()
    logs = {'recerr_sq':.0, 'kldiv':.0, 'unmix':.0, 'dataug':.0, 'lact_dec':.0}

    for batch_idx, (data,) in enumerate(loader):
        data = data.to(device)
        batch_size = len(data)
        optimizer.zero_grad()

        # inference & reconstruction on original data
        z_phy_stat, z_aux2_stat, unmixed = model.encode(data)
        z_phy, z_aux2 = model.draw(z_phy_stat,  z_aux2_stat, hard_z=False)
        x_PB, x_P, x_lnvar, y = model.decode(z_phy, z_aux2, full=True)
        x_var = torch.exp(x_lnvar)

        if args.rescale:
            data_rescaled, x_PB_rescaled = rescale(data, x_PB)
        else:
            data_rescaled = data; x_PB_rescaled = x_PB

        # ELBO
        recerr_sq, kldiv = loss_function(args, data_rescaled, z_phy_stat, z_aux2_stat, x_PB_rescaled)

        # unmixing regularization (R_{DA,1})
        if not args.no_phy:
            reg_unmix = torch.sum((unmixed - y.detach()).pow(2), dim=[1,2,3]).mean()
        else:
            reg_unmix = torch.zeros(1, device=device).squeeze()

        # data augmentation regularization (R_{DA,2})
        if not args.no_phy:
            model.eval()
            with torch.no_grad():
                aug_z_phy = torch.cat([
                    torch.rand((batch_size,1), device=device)*(args.range_I0[1]-args.range_I0[0])+args.range_I0[0],
                    torch.rand((batch_size,1), device=device)*(args.range_A[1]-args.range_A[0])+args.range_A[0],
                    torch.rand((batch_size,1), device=device)*(args.range_e[1]-args.range_e[0])+args.range_e[0],
                    torch.rand((batch_size,1), device=device)*(args.range_theta[1]-args.range_theta[0])+args.range_theta[0]], dim=1)
                aug_y = model.generate_physonly(aug_z_phy)
            model.train()
            aug_feature_phy = model.enc.func_feat(aug_y.detach().expand(batch_size,3,69,69))
            aug_infer = model.enc.func_z_phy_mean(aug_feature_phy)
            reg_dataug = (aug_infer - aug_z_phy).pow(2).mean()
        else:
            reg_dataug = torch.zeros(1, device=device).squeeze()

        # least action principle (R_ppc)
        if not args.no_phy:
            reg_lact_dec = torch.sum((x_PB - x_P).pow(2), dim=[1,2,3]).mean()
        else:
            reg_lact_dec = torch.zeros(1, device=device).squeeze()

        # loss function
        kldiv_balanced = (args.balance_kld + args.balance_lact_enc) * kldiv * x_var.detach()
        if not args.no_phy and epoch < args.epochs_pretrain:
            loss = args.balance_dataug*reg_dataug
        else:
            loss = recerr_sq + kldiv_balanced \
                + args.balance_unmix*reg_unmix + args.balance_dataug*reg_dataug + args.balance_lact_dec*reg_lact_dec

        # update model parameters
        loss.backward()

        if args.grad_clip>0.0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()

        # log
        logs['recerr_sq'] += recerr_sq.detach()*batch_size
        logs['kldiv'] += kldiv.detach()*batch_size
        logs['unmix'] += reg_unmix.detach()*batch_size
        logs['dataug'] += reg_dataug.detach()*batch_size
        logs['lact_dec'] += reg_lact_dec.detach()*batch_size

    for key in logs:
        logs[key] /= len(loader.dataset)
    print('====> Epoch: {}  Training (rec. err.)^2: {:.4f}  kldiv: {:.4f}  unmix: {:4f}  dataug: {:4f}  lact_dec: {:4f}'.format(
        epoch, logs['recerr_sq'], logs['kldiv'], logs['unmix'], logs['dataug'], logs['lact_dec']))
    return logs


def valid(epoch, args, device, loader, model):
    model.eval()
    logs = {'recerr_sq':.0, 'kldiv':.0}
    with torch.no_grad():
        for i, (data,) in enumerate(loader):
            data = data.to(device)
            batch_size = len(data)
            z_phy_stat, z_aux2_stat, x, _ = model(data)

            if args.rescale:
                data, x = rescale(data, x)
            recerr_sq, kldiv = loss_function(args, data, z_phy_stat, z_aux2_stat, x)

            logs['recerr_sq'] += recerr_sq.detach()*batch_size
            logs['kldiv'] += kldiv.detach()*batch_size

    for key in logs:
        logs[key] /= len(loader.dataset)
    print('====> Epoch: {}  Validation (rec. err.)^2: {:.4f}  kldiv: {:.4f}'.format(
        epoch, logs['recerr_sq'], logs['kldiv']))
    return logs


if __name__ == '__main__':

    parser = set_parser()
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")


    # set random seed
    torch.manual_seed(args.seed)


    # load training/validation data
    data_all = np.load('{}/data_all.npy'.format(args.datadir))


    # train/valid/test split
    idx = torch.randperm(data_all.shape[0]).numpy().astype(np.intp)
    idx_train = idx[0:args.train_size]
    idx_valid = idx[args.train_size:args.train_size+args.valid_size]
    np.savetxt('{}/idx_train.txt'.format(args.outdir), idx_train, fmt='%d')
    np.savetxt('{}/idx_valid.txt'.format(args.outdir), idx_valid, fmt='%d')
    np.savetxt('{}/idx_test.txt'.format(args.outdir), idx[args.train_size+args.valid_size:], fmt='%d')

    data_train = data_all[idx_train]
    data_valid = data_all[idx_valid]

    data_train_tensor = torch.Tensor(data_train).float()
    data_valid_tensor = torch.Tensor(data_valid).float()


    # data augmentation
    if args.data_augmentation > 1:
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(360, fillcolor=0.0),
            # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.2)
        ])
        data_train_tensor_original = data_train_tensor.clone()
        data_valid_tensor_original = data_valid_tensor.clone()
        for i in range(args.data_augmentation-1):
            data_train_tensor = torch.cat([data_train_tensor, trans(data_train_tensor_original)], dim=0)
            data_valid_tensor = torch.cat([data_valid_tensor, trans(data_valid_tensor_original)], dim=0)


    # set data loaders
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    loader_train = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_train_tensor),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    loader_valid = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_valid_tensor),
        batch_size=args.batch_size, shuffle=False, **kwargs)


    # set model
    model = VAE(vars(args)).to(device)


    # set optimizer
    kwargs = {'lr': args.learning_rate, 'weight_decay': args.weight_decay, 'eps': args.adam_eps}
    optimizer = optim.Adam(model.parameters(), **kwargs)


    print('start training with device', device)
    print(vars(args))
    print()


    # save args
    with open('{}/args.json'.format(args.outdir), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


    # create log files
    with open('{}/log.txt'.format(args.outdir), 'w') as f:
        print('# epoch recerr_sq kldiv unmix dataug lact_dec valid_recerr_sq valid_kldiv duration', file=f)


    # main iteration
    info = {'bestvalid_epoch':0, 'bestvalid_recerr':1e10}
    dur_total = .0
    for epoch in range(1, args.epochs + 1):
        # training
        start_time = time.time()
        logs_train = train(epoch, args, device, loader_train, model, optimizer)
        dur_total += time.time() - start_time

        # validation
        logs_valid = valid(epoch, args, device, loader_valid, model)

        # save loss information
        with open('{}/log.txt'.format(args.outdir), 'a') as f:
            print('{} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e}'.format(epoch,
                logs_train['recerr_sq'], logs_train['kldiv'], logs_train['unmix'], logs_train['dataug'], logs_train['lact_dec'],
                logs_valid['recerr_sq'], logs_valid['kldiv'], dur_total), file=f)

        # save model if best validation loss is achieved
        if logs_valid['recerr_sq'] < info['bestvalid_recerr']:
            info['bestvalid_epoch'] = epoch
            info['bestvalid_recerr'] = logs_valid['recerr_sq']
            torch.save(model.state_dict(), '{}/model.pt'.format(args.outdir))
            print('best model saved')

        # save model at interval
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), '{}/model_e{}.pt'.format(args.outdir, epoch))

        print()

    print()
    print('end training')
