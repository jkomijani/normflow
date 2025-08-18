# Auto Regressive Model
# Javad Komijani, 13/Oct/2024, updated 18/Aug/2025

"""
This module implements an auto-regressive model for high-dimensional lattice
systems, designed to efficiently handle large lattices by leveraging an
exponentially decreasing number of transformation layers.

The model is built as a sequence of submodules, each responsible for
transforming a single active slice of the lattice while conditioning on a
mixture of all previously processed slices (frozen variables). This design
ensures scalability and stability when modeling large structured data.

To run the main function with default options, use:

    >>> python3 $filename

For parallel training, e.g., with 2 nodes and 4 processors per node, use:

    >>> torchrun --nproc_per_node=4 $filename --world_size 8

The `world_size` option serves two purposes:

1. Dividing the batch size.
2. Running `execute_ddp_training` if `world_size > 1`.
"""

# pylint: disable=arguments-differ, too-many-locals
# pylint: disable=too-many-arguments, too-many-positional-arguments

from functools import partial
from typing import Tuple

import math
import torch
import normflow

from normflow import Model
from normflow.prior import NormalPrior
from normflow.action import ScalarPhi4Action

from normflow.mask import (
    EvenOddMask,
    ListPartitioner
)

from normflow.nn import (
    Module_,
    DistConvertor_,
    AffineCoupling_,
    ConvBlock
)

from normflow.nn import (
    FibonacciTilingSplitter,
    AutoRegModule_
)


# =============================================================================
def main(
    # Lattice setup
    kappa: float = 0.67,
    m_sq: float = -4*0.67,
    lambd: float = 0.5,
    lat_shape: tuple = (8, 8),
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
    """The main file for building and training the model."""

    if debug:
        torch.manual_seed(213)

    net_ = assemble_fibo_autoreg_module(lat_shape=lat_shape, **net_kwargs)
    action = ScalarPhi4Action(kappa=kappa, m_sq=m_sq, lambd=lambd)
    prior = NormalPrior(shape=lat_shape)

    model = Model(net_=net_, prior=prior, action=action)

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
def assemble_fibo_autoreg_module(
    lat_shape: Tuple[int], knots_len=50, hidden_sizes: Tuple[int] = (8,)
):
    """
    An auto-regressive model suitable for large lattices because of its
    exponential decrease in number of layers.

    There is a list of submodules, each of which
    takes one slice as an active slice and a mixture of all previous slices as
    a frozen variable and transforms the active slice. The list of submodules
    can be provided as an option called `nets_`, otherwise the default choice
    will be used.

    Parameters
    ----------
    lat_shape: Tuple[int]
        shape of underlying lattice
    nets_: Tuple[Callable] | None, optional
        a tuple of submodules or None. The submodules, if provided, must be
        instances of `Module_`. Default is None, indicating a default set of
        submodules are used; the submoduels are a combinations of
        `DistConvetor_` and `RQSplineCoupling_`.
    """
    splitter = FibonacciTilingSplitter(lat_shape)

    metadata_list = splitter.metadata_list

    conv_kwargs = {
        'in_channels': 1,
        'out_channels': 2,
        'kernel_size': 3,
        'conv_ndim': len(lat_shape),
        'hidden_sizes': hidden_sizes,
        'acts': (*[torch.nn.LeakyReLU()]*len(hidden_sizes), None)
    }

    nets_ = [None] * len(metadata_list)

    for ind, (_, _, _, a_shape, _) in enumerate(metadata_list):
        if ind == 0:
            nets_[0] = DistConvertor_(knots_len)
        else:
            nets1 = [ConvBlock(**conv_kwargs) for _ in range(4)]
            nets2 = [ConvBlock(**conv_kwargs) for _ in range(4)]
            nets_[ind] = AutoRegSubmodule_(nets1, nets2, a_shape)

    return AutoRegModule_(nets_, splitter=splitter)


# =============================================================================
class AutoRegSubmodule_(Module_):  # pylint: disable=invalid-name
    """
    A building block (submodule) for autoregressive models, designed to
    work with `FibonacciTilingSplitter`.

    This submodule operates on a pair of variables:
    - **active variables**: the current portion of the data being updated.
    - **frozen variables**: all previously updated portions that serve as
      conditioning context.

    The forward pass transforms the active variables through coupling
    layers and a distribution converter, then recombines them with the
    frozen variables. The resulting pair can be passed as input to the
    next autoregressive submodule. The reverse pass exactly inverts this
    sequence.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the active and frozen variables.
    nets1 : Tuple[Module, ...]
        Networks for the first affine coupling layer. Each operates on
        both active and frozen variables. These are interwoven with
        `None` to indicate that frozen variables remain unchanged even
        when it's their "turn" in the sequence.
    nets2 : Tuple[Module, ...]
        Networks for the second affine coupling layer. These act only
        on the transformed active variables, using an even/odd mask.
    knots_len : int, optional
        Number of knots in the `DistConvertor_` spline that operates on
        the transformed active variables. Default is 10.

    Notes
    -----
    - `affine1_` conditions on both active and frozen variables, but
      interleaved `None` entries ensure frozen variables are never
      updated directly.
    - `affine2_` further transforms active variables with an even/odd
      masking strategy.
    - `dc_` applies a smooth invertible distribution transformation to
      the final active variables.
    """

    def __init__(self, nets1, nets2, shape, knots_len=10):
        super().__init__()

        # Smooth distribution converter for transformed active variables
        self.dc_ = DistConvertor_(knots_len, smooth=True)

        # First affine coupling: conditioned on both active + frozen variables
        mask1 = ListPartitioner()
        # Interleave None values between nets1 modules
        nets1 = [item for net in nets1 for item in (net, None)]
        self.affine1_ = AffineCoupling_(nets1, mask=mask1)

        # Second affine coupling: conditioned on even/odd partition of input
        mask2 = EvenOddMask(shape=shape)
        self.affine2_ = AffineCoupling_(nets2, mask=mask2)

    def forward(self, x, log0=0):
        """
        Forward pass through the submodule.

        Parameters
        ----------
        x : Tuple[Tensor, Tensor]
            Pair of (x_active, x_frozen).
        log0 : Tensor or float, optional
            Initial log-determinant of the Jacobian.

        Returns
        -------
        x : Tuple[Tensor, Tensor]
            Updated (x_active, x_frozen) pair.
        logj : Tensor
            Accumulated log-determinant of the Jacobian.
        """
        x_active, x_frozen = x

        # First affine coupling (depends on both active + frozen)
        (x_active, _), logj = self.affine1_(x, log0=log0)

        # Second affine coupling (even/odd mask only)
        x_active, logj = self.affine2_(x_active, log0=logj)

        # Smooth invertible distribution transform
        x_active, logj = self.dc_(x_active, log0=logj)

        # Recombine updated active with frozen for the next submodule
        x = (x_active, x_frozen)
        return x, logj

    def reverse(self, x, log0=0):
        """
        Reverse pass (inverse transformation).

        Parameters
        ----------
        x : Tuple[Tensor, Tensor]
            Pair of (x_active, x_frozen).
        log0 : Tensor or float, optional
            Initial log-determinant of the Jacobian.

        Returns
        -------
        x : Tuple[Tensor, Tensor]
            Reconstructed (x_active, x_frozen) pair.
        logj : Tensor
            Accumulated log-determinant of the Jacobian.
        """
        x_active, x_frozen = x

        # Inverse distribution transformation
        x_active, logj = self.dc_.reverse(x_active, log0=log0)

        # Inverse second affine coupling
        x_active, logj = self.affine2_.reverse(x_active, log0=logj)

        # Inverse first affine coupling (reintegrates frozen vars)
        (x_active, _), logj = self.affine1_.reverse(
            (x_active, x_frozen), log0=logj
        )

        x = (x_active, x_frozen)
        return x, logj


# =============================================================================
def _unittest(rel_tol=1e-1):
    # The reference point `loss_ref` is obtained on GPU with double precision.
    # Results vary between CPU and GPU, that's why rel_tol is so large!
    model = main(debug=True, n_epochs=5, print_every=None)
    loss = model.trainer.compute_metrics(batch_size=16)[0]
    loss_ref = -44.631146973713
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
    add("--knots_len", dest="knots_len", type=int)
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
