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

from typing import Tuple
from functools import partial

import math
import torch
import normflow

from torch.nn import BatchNorm2d

from normflow import Model
from normflow.prior import NormalPrior
from normflow.action import ScalarPhi4Action
from normflow.mask import EvenOddMask

from normflow.nn import (
    ModuleList_,
    DistConvertor_,
    make_psd_block,
    AffineCoupling_,
    AvgNeighborPool,
    ConvBlock
)


# =============================================================================
def main(
    # Lattice setup
    kappa: float = 0.67,
    m_sq: float = -4 * 0.67,
    lambd: float = 0.5,
    lat_shape: Tuple[int, ...] = (8, 8),
    # Training setup
    n_epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 0.01,
    path_gradient_autodiff: bool = True,
    alpha_tmax: bool = None,
    world_size: int = 1,
    print_every: int = 100,
    print_bsize: int | None = None,
    # IO & test
    load_fname: str = None,
    save_fname: str = None,
    debug: bool = False,
    # Architecture setup
    **net_kwargs
):
    """
    Build, configure, and train a lattice model.

    This function assembles the network, sets up the action and prior, and
    runs training using either single-GPU or DDP mode.

    Steps:
        1. Assemble the network with `assemble_net`.
        2. Define the lattice action (`ScalarPhi4Action`) and prior
           distribution (`NormalPrior`).
        3. Wrap into a `Model` and configure parameter groups.
        4. Set up optimizer scheduler and training arguments.
        5. Execute training (DDP if `world_size > 1`) and perform checks.

    Args:
        kappa, m_sq, lambd: Lattice action parameters.
        lat_shape: Shape of the lattice.
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        path_gradient_autodiff: Whether to use path-wise gradient autodiff.
        alpha_tmax: Optional alpha tmax for scheduler.
        world_size: Number of parallel workers for DDP.
        print_every: Steps between console prints.
        print_bsize: Optional batch size for printing metrics.
        load_fname: Path to load checkpoint.
        save_fname: Path to save checkpoint.
        debug: If True, sets a fixed random seed for reproducibility.
        **net_kwargs: Additional keyword arguments for `assemble_net`.

    Returns:
        Model: The trained model instance.
    """

    if debug:
        torch.manual_seed(213)

    net_ = assemble_net(lat_shape=lat_shape, **net_kwargs)
    action = ScalarPhi4Action(kappa=kappa, m_sq=m_sq, lambd=lambd)
    prior = NormalPrior(shape=lat_shape)

    model = Model(net_=net_, prior=prior, action=action)

    # Training setup
    model.net_.setup_groups(
        groups=[
            {'ind': [0, 1, 3], 'hyper': {'weight_decay': 1e-4}},
            {'ind': [2], 'hyper': {'weight_decay': 1e-2}}
        ]
    )

    checkpoint_dict = {
        'print_every': print_every,
        'print_bsize': print_bsize and print_bsize // world_size
    }

    scheduler = partial(
        torch.optim.lr_scheduler.CosineAnnealingLR,
        T_max=int(1.01 * n_epochs + 1)
    )

    train_kwargs = {
        'n_epochs': n_epochs,
        'batch_size': batch_size // world_size,
        'path_gradient_autodiff': path_gradient_autodiff,
        'load_checkpoint_path': load_fname,
        'save_checkpoint_path': save_fname,
        'scheduler': scheduler,
        'alpha_tmax': alpha_tmax,
        'hyperparam': {'lr': lr},
        'checkpoint_dict': checkpoint_dict
    }

    if world_size > 1:
        model.execute_ddp_training(**train_kwargs)
    else:
        print("number of model parameters =", model.net_.npar)
        model.train(**train_kwargs)
        normflow.reverse_flow_sanitychecker(model)

    return model


# =============================================================================
def assemble_net(
    lat_shape: Tuple[int, ...],
    n_layers: int = 4,
    hidden_sizes: Tuple[int, ...] = (8, 8),
    zee2sym: bool = True,
    acts: Tuple[torch.nn.Module, ...] | None = None,
    len0: int = 4,
    len1: int = 10,
    len2: int = 50,
    len3: int = 50
):
    """
    Assemble a modular neural network for lattice data as a `ModuleList_`.

    The network includes, in order:
        1. PSD block (mean-field + FFT-based) for lattice modes.
        2. Optional DistConvertor_ for intermediate activation.
        3. Affine coupling blocks (ConvBlock inside AffineCoupling_).
        4. Optional DistConvertor_ for output transformation.

    Args:
        lat_shape: Shape of the lattice input.
        n_layers: Number of affine layers in each affine coupling.
        hidden_sizes: Hidden channel sizes for ConvBlock in an affine layer.
        zee2sym: If True, enforces Z2 symmetry for activations and converters.
        acts: Optional activations for ConvBlocks; defaults to Tanh (Z2) or
            LeakyReLU.
        len0: Reserved for number of layers in PSD block mean-field.
        len1: Number of spline knots in the PSD block (ipsd_knots_len).
        len2: Size of first DistConvertor_ (optional intermediate activation).
        len3: Size of final DistConvertor_ (optional output activation).

    Returns:
        ModuleList_: List of modules forming the complete lattice network.
    """

    # 1. PSD block
    psd_block_ = make_psd_block(
        lat_shape, meanfield_n_layers=len0, ipsd_knots_len=len1
    )

    nets_list = [psd_block_]

    # 2. include (possible) activation
    if len2 > 1:
        nets_list.append(
            DistConvertor_(len2, symmetric=zee2sym, smooth=True)
        )

    # 3. Add (possible) affine blocks
    if acts is None:
        act = torch.nn.Tanh() if zee2sym else torch.nn.LeakyReLU()
        acts = (*[act]*len(hidden_sizes), None)

    norms = (
        *[BatchNorm2d(n, affine=not zee2sym) for n in hidden_sizes],
        None
    )

    conv_dict = {
        'in_channels': 1,
        'out_channels': 2,
        'hidden_sizes': hidden_sizes,
        'kernel_size': 3,
        'padding_mode': 'circular',
        'conv_ndim': len(lat_shape),
        'acts': acts,
        'norms': norms,
        'pre_act': AvgNeighborPool(),
        'bias': not zee2sym
    }

    mask = EvenOddMask(shape=lat_shape)

    nets_list.append(
        AffineCoupling_(
            [ConvBlock(**conv_dict) for _ in range(n_layers)],
            mask=mask
        )
    )

    # 4. include (possible) activation
    if len3 > 1:
        nets_list.append(
            DistConvertor_(len3, symmetric=zee2sym, smooth=True)
        )

    return ModuleList_(nets_list)


# =============================================================================
def _unittest(rel_tol: float = 1e-1):
    """
    Minimal unit test for the PSD + affine coupling model.

    Due to CPU/GPU differences, the relative tolerance is intentionally large.

    Args:
        rel_tol: Relative tolerance for comparing computed loss to reference.

    Returns:
        bool: True if the computed loss is within tolerance, False otherwise.
    """
    model = main(debug=True, n_epochs=5, print_every=None)
    loss = model.trainer.compute_metrics(batch_size=16)[0]
    loss_ref = -54.615462066452416
    passed = math.isclose(loss, loss_ref, rel_tol=rel_tol)
    if not passed:
        print(f"Unittest Failed in psd_affine_coupling: {loss} != {loss_ref}")
    return passed


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add = parser.add_argument

    # Lattice setup
    add("--lat_shape", dest="lat_shape", type=int, nargs='+')
    add("--m_sq", dest="m_sq", type=float)
    add("--lambd", dest="lambd", type=float)
    add("--kappa", dest="kappa", type=float)
    # Architecture setup
    add("--n_layers", dest="n_layers", type=int)
    add("--len0", dest="len0", type=int)
    add("--len1", dest="len1", type=int)
    add("--len2", dest="len2", type=int)
    add("--len3", dest="len3", type=int)
    add("--zee2sym", dest="zee2sym", type=bool)
    add("--hidden_sizes", dest="hidden_sizes", type=int, nargs='+')
    # Training setup
    add("--batch_size", dest="batch_size", type=int)
    add("--lr", dest="lr", type=float)
    add("--n_epochs", dest="n_epochs", type=int)
    add("--world_size", dest="world_size", type=int)
    add("--path_gradient_autodiff", dest="path_gradient_autodiff", type=bool)
    add("--alpha_tmax", dest="alpha_tmax", type=int)
    add("--print_every", dest="print_every", type=int)
    add("--print_bsize", dest="print_bsize", type=int)
    # IO & test
    add("--load_fname", dest="load_fname", type=str)
    add("--save_fname", dest="save_fname", type=str)
    add("--unittest", dest="unittest", type=bool)

    args = vars(parser.parse_args())
    args = {key: value for key, value in args.items() if value is not None}

    if "unittest" in args.keys():
        _unittest()
    else:
        main(**args)
