# Javad Komijani, 2021-2025

"""
This file implements a model similar to the one defined in [arXiv:2301.01504]
but without PSD flow.

To run the main function with default options, use:

    >>> python3 $filename

For parallel training, e.g., with 2 nodes and 4 processors per node, use:

    >>> torchrun --nproc_per_node=4 $filename
"""

from functools import partial

import torch
import normflow

from torch.optim.lr_scheduler import CosineAnnealingLR

from normflow import Model
from normflow.prior import NormalPrior
from normflow.action import ScalarPhi4Action
from normflow.mask import EvenOddMask

from normflow.nn import (
    ModuleList_,
    AffineCoupling_,
    ConvBlock
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
    # IO & test
    log_name: str = None,
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
        3. Wrap into a `Model`.
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
        log_name: Name for the training logger.
        load_fname: Path to load checkpoint.
        save_fname: Path to save checkpoint.
        debug: If True, sets a fixed random seed for reproducibility.
        **net_kwargs: Additional keyword arguments for `assemble_net`.

    Returns:
        Model: The trained model instance.
    """

    if debug:
        torch.manual_seed(42)

    net_ = assemble_net(lat_shape=lat_shape, **net_kwargs)
    action = ScalarPhi4Action(kappa=kappa, m_sq=m_sq, lambd=lambd)
    prior = NormalPrior(shape=lat_shape)

    model = Model(network_fn_=net_, prior=prior, action=action)

    training_config = {
        'hyperparam': {'lr': lr},
        'lr_scheduler_class': partial(CosineAnnealingLR, T_max=1 + n_epochs),
        'path_gradient_autodiff': path_gradient_autodiff,
        'alpha_tmax': alpha_tmax,
        'log_name': log_name,
        'load_checkpoint_path': load_fname,
        'save_checkpoint_path': save_fname
    }

    world_size = model.trainer.device_handler.world_size

    if world_size > 1:
        seeds_list = torch.randint(2**32 - 1, size=(world_size,)).tolist()
        training_config["seeds_list"] = seeds_list

    model.trainer.run_training(n_epochs, batch_size, **training_config)

    if world_size == 1:
        if n_epochs > 0:
            log_dict = model.trainer.logger.load_numpy()
            print("mean(loss[-100:])", log_dict['loss'][-100:].mean())

            ess = model.compute_metrics(batch_size=batch_size)[0]
            print(f"\nESS: {ess}\n")

        normflow.reverse_flow_sanitychecker(model)

    return model


# =============================================================================
def assemble_net(
    *, lat_shape,
    n_layers=16,
    hidden_sizes=(8, 8),
    zee2sym=True,
    acts=None,
):
    """Assemble a module and return it as an instance of `ModuleList_`."""

    nets_list = []

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

    return ModuleList_(nets_list)


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add = parser.add_argument

    # Lattice setup
    add("--lat_shape", type=int, nargs='+')
    add("--m_sq", type=float)
    add("--lambd", type=float)
    add("--kappa", type=float)
    # Architecture setup
    add("--n_layers", type=int)
    add("--zee2sym", type=bool)
    add("--hidden_sizes", type=int, nargs='+')
    # Training setup
    add("--batch_size", type=int)
    add("--lr", type=float)
    add("--n_epochs", type=int)
    add("--path_gradient_autodiff", type=bool)
    add("--alpha_tmax", type=int)
    # IO & test
    add("--log_name", type=str)
    add("--load_fname", type=str)
    add("--save_fname", type=str)

    args = vars(parser.parse_args())
    args = {key: value for key, value in args.items() if value is not None}

    main(**args)
