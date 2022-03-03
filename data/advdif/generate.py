"""Generate data from advection-diffusion equation.
"""

import argparse
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import convolve1d


def generate_traj(init_y, dcoeff, ccoeff, dx, dt, len_episode):
    def fun(t, y):
        y_x = convolve1d(y, weights=[-1.0, 0.0, 1.0], mode='constant', cval=0.0)
        y_xx = convolve1d(y, weights=[1.0, -2.0, 1.0], mode='constant', cval=0.0)
        return dcoeff*y_xx/dx/dx - ccoeff*y_x/2.0/dx
    sol = solve_ivp(fun, (0.0, dt*(len_episode-1)), init_y, dense_output=True, method='DOP853')
    t = np.linspace(0.0, dt*(len_episode-1), len_episode)
    return t, sol.sol(t)


def generate_data(rng, n_samples, range_init_mag, range_dcoeff, range_ccoeff,
                  dx, dt, n_grids, len_episode, noise_std):
    assert range_init_mag[0] <= range_init_mag[1]
    init_mags = rng.uniform(low=range_init_mag[0], high=range_init_mag[1], size=n_samples)
    assert range_dcoeff[0] <= range_dcoeff[1]
    dcoeffs = rng.uniform(low=range_dcoeff[0], high=range_dcoeff[1], size=n_samples)
    assert range_ccoeff[0] <= range_ccoeff[1]
    ccoeffs = rng.uniform(low=range_ccoeff[0], high=range_ccoeff[1], size=n_samples)

    # solve ODE
    x_grid = np.linspace(0.0, dx*(n_grids-1), n_grids)
    init_y_base = np.sin(x_grid / x_grid[-1] * np.pi)
    x = np.empty((n_samples, n_grids, len_episode))
    for i in range(n_samples):
        init_y_i = init_y_base * init_mags[i]
        t, x_i = generate_traj(init_y_i, dcoeffs[i], ccoeffs[i], dx, dt, len_episode)
        x[i] = x_i

    # observation noise
    x = x + rng.normal(loc=0.0, scale=noise_std, size=x.shape)

    return t, x, init_mags, dcoeffs, ccoeffs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # output
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)

    # configurations
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--n-grids', type=int, default=20)
    parser.add_argument('--len-episode', type=int, default=50)
    parser.add_argument('--dx', type=float, default=0.15707963267) # pi/20
    parser.add_argument('--dt', type=float, default=0.12)
    parser.add_argument('--range-init-mag', type=float, nargs=2, default=[0.5, 1.5])
    parser.add_argument('--range-dcoeff', type=float, nargs=2, default=[1e-2, 1e-1])
    parser.add_argument('--range-ccoeff', type=float, nargs=2, default=[1e-2, 1e-1])
    parser.add_argument('--noise-std', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1234567890)
    args = parser.parse_args()

    # check inputs
    assert args.range_init_mag[0] <= args.range_init_mag[1]
    assert args.range_dcoeff[0] <= args.range_dcoeff[1]
    assert args.range_ccoeff[0] <= args.range_ccoeff[1]


    # set random seed
    rng = np.random.default_rng(args.seed)


    # generate data
    kwargs = {'range_init_mag': args.range_init_mag,
              'range_dcoeff': args.range_dcoeff, 'range_ccoeff': args.range_ccoeff,
              'dx': args.dx, 'dt': args.dt, 'n_grids':args.n_grids, 'len_episode':args.len_episode,
              'noise_std':args.noise_std}
    t, data, init_mags, dcoeffs, ccoeffs = generate_data(rng, args.n_samples, **kwargs)


    # save args
    with open('{}/args_{}.json'.format(args.outdir, args.name), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


    # save data
    np.save('{}/data_{}.npy'.format(args.outdir, args.name), data)
    # np.savetxt('{}/data_{}_0.txt'.format(args.outdir, args.name), data[0])
    print('saved data: min(abs(x))={:.3e}, max(abs(x))={:.3e}'.format(
        np.min(np.abs(data)), np.max(np.abs(data)) ))


    # save true parameters
    np.savetxt('{}/true_params_{}.txt'.format(args.outdir, args.name),
        np.stack([init_mags, dcoeffs, ccoeffs], axis=1), fmt='%.9e',
        header='init_mag dcoeff ccoeff')
    print('saved true parameters')
