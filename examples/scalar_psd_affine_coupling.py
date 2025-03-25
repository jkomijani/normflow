# Javad Komijani, 2021-2025

"""
This file implements a model similar to the one defined in [arXiv:2301.01504]
with PSD flow and coupling layers.

To run the main function with default options, use:

    >>> python3 $filename

For parallel training, e.g., with 2 nodes and 4 processors per node, use:

    >>> torchrun --nproc_per_node=4 $filename --world_size 8

The `world_size` option serves two purposes:

1. Dividing the batch size.
2. Running `execute_ddp_training` if `world_size > 1`.
"""

import math

from functools import partial

import torch

from normflow import Model
from normflow.prior import NormalPrior
from normflow.action import ScalarPhi4Action
from normflow.mask import EvenOddMask
from normflow.nn import ModuleList_, Identity_, DistConvertor_, AffineCoupling_
from normflow.nn import FFTNet_, MeanFieldNet_, PSDBlock_
from normflow.nn import ConvBlock


# =============================================================================
def main(
    kappa: float = 0.67,
    m_sq: float = -4*0.67,
    lambd: float = 0.5,
    lat_shape: tuple = (8, 8),
    n_epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 0.01,
    load_fname: str = None,
    save_fname: str = None,
    world_size: int = 1,
    print_every: int = 100,
    debug: bool = False,
    **net_kwargs
):
    """The main file for building and training the model."""

    if debug:
        torch.manual_seed(213)

    net_ = assemble_net(lat_shape=lat_shape, **net_kwargs)
    action = ScalarPhi4Action(kappa=kappa, m_sq=m_sq, lambd=lambd)
    prior = NormalPrior(shape=lat_shape)

    model = Model(net_=net_, prior=prior, action=action)
    model.trainer.path_gradient_autodiff = True

    # print("number of model parameters =", model.net_.npar)

    model.net_.setup_groups(
        groups=[
            {'ind': [0, 1, 3], 'hyper': {'weight_decay': 1e-4}},
            {'ind': [2], 'hyper': {'weight_decay': 1e-2}}
        ]
    )

    if load_fname is not None:
        model.load_checkpoint(load_fname)

    scheduler = partial(
        torch.optim.lr_scheduler.CosineAnnealingLR, T_max=int(1.01 * n_epochs)
    )

    train_kwargs = {
        'n_epochs': n_epochs,
        'batch_size': batch_size // world_size,
        'scheduler': scheduler,
        'hyperparam': {'lr': lr},
        'checkpoint_dict': {'print_every': print_every}
    }

    if world_size > 1:
        if debug:
            train_kwargs.update({'seeds_list': range(world_size)})
        model.execute_ddp_training(**train_kwargs)
    else:
        model.train(**train_kwargs)

    if save_fname is not None:
        model.save_checkpoint(save_fname)

    return model


# =============================================================================
def assemble_net(
    *, lat_shape,
    n_layers=4,
    hidden_sizes=(8, 8),
    zee2sym=True,
    acts=None,
    knots0_len=10,
    knots1_len=10,
    knots2_len=50,
    knots4_len=50
):
    """Assemble a module and return it as an instance of `ModuleList_`."""

    mfdict = dict(
        knots_len=knots0_len, symmetric=zee2sym, final_scale=True, smooth=True
    )

    fftdict = dict(knots_len=knots1_len, ignore_zeromode=True)

    nets_list = []

    # 1. First block
    mfnet_ = MeanFieldNet_.build(**mfdict) if (knots0_len > 1) else Identity_()
    fftnet_ = FFTNet_.build(lat_shape, **fftdict)
    nets_list.append(PSDBlock_(mfnet_=mfnet_, fftnet_=fftnet_))

    # 2. include (possible) activation
    if knots2_len > 1:
        nets_list.append(
            DistConvertor_(knots2_len, symmetric=zee2sym, smooth=True)
        )

    # 3. Add (possible) affine blocks
    if acts is None:
        act = torch.nn.Tanh() if zee2sym else torch.nn.LeakyReLU()
        acts = (*[act]*len(hidden_sizes), None)

    conv_dict = dict(
        in_channels=1,
        out_channels=2,
        hidden_sizes=hidden_sizes,
        kernel_size=3,
        padding_mode='circular',
        conv_ndim=len(lat_shape),
        acts=acts,
        bias=not zee2sym
    )

    mask = EvenOddMask(shape=lat_shape)

    nets_list.append(
        AffineCoupling_(
            [ConvBlock(**conv_dict) for _ in range(n_layers)],
            mask=mask
        )
    )

    # 4. include (possible) activation
    if knots4_len > 1:
        nets_list.append(
            DistConvertor_(knots4_len, symmetric=zee2sym, smooth=True)
        )

    return ModuleList_(nets_list)


def _unittest(rel_tol=1e-1):
    # results vary between CPU and GPU, that's why rel_tol is so large!
    model = main(debug=True, n_epochs=5, print_every=None)
    loss = model.trainer.compute_metrics(batch_size=16)[0]
    passed = math.isclose(loss, -52.72515650, rel_tol=rel_tol)
    if not passed:
        print(f"Unittest Failed: {loss} != -52.72515650")
    return passed


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add = parser.add_argument

    add("--lat_shape", dest="lat_shape", type=int, nargs='+')
    add("--m_sq", dest="m_sq", type=float)
    add("--lambd", dest="lambd", type=float)
    add("--kappa", dest="kappa", type=float)
    add("--lr", dest="lr", type=float)
    add("--knots0_len", dest="knots0_len", type=int)
    add("--knots1_len", dest="knots1_len", type=int)
    add("--knots2_len", dest="knots2_len", type=int)
    add("--knots4_len", dest="knots4_len", type=int)
    add("--zee2sym", dest="zee2sym", type=bool)
    add("--n_layers", dest="n_layers", type=int)
    add("--batch_size", dest="batch_size", type=int)
    add("--n_epochs", dest="n_epochs", type=int)
    add("--hidden_sizes", dest="hidden_sizes", type=int, nargs='+')
    add("--world_size", dest="world_size", type=int)
    add("--load_fname", dest="load_fname", type=str)
    add("--save_fname", dest="save_fname", type=str)
    add("--unittest", dest="unittest", type=bool)

    args = vars(parser.parse_args())
    args = {key: value for key, value in args.items() if value is not None}

    if "unittest" in args.keys():
        _unittest()
    else:
        main(**args)
