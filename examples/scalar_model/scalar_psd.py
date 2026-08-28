# Javad Komijani, 2021-2026

"""This file implements a PSD flow."""

from typing import Tuple
from functools import partial

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import normflow

from normflow import Model
from normflow.prior import NormalPrior
from normflow.action import ScalarPhi4Action

from normflow.nn import (
    ModuleList_,
    make_real_line_rqs,
    make_psd_block,
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

    # model.trainer.device_handler.training_device = 'cpu'
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
    lat_shape: Tuple[int, ...],
    len0: int = 4,
    num_spline_knots1: int = 10,
    num_spline_knots2: int = 50,
):
    """
    Assemble a modular neural network for lattice data as a `ModuleList_`.

    The network includes, in order:
        1. PSD block (mean-field + FFT-based) for lattice modes.
        2. Optional `make_real_line_rqs` for output transformation.

    Args:
        lat_shape: Shape of the lattice input.
        len0: Reserved for number of layers in PSD block mean-field.
        num_spline_knots1: Number of spline knots in the PSD block.
        num_spline_knots2: For `make_real_line_rqs` (output activation).

    Returns:
        ModuleList_: List of modules forming the complete lattice network.
    """
    # 1. PSD block
    psd_block_ = make_psd_block(
        lat_shape, meanfield_n_layers=len0, ipsd_knots_len=num_spline_knots1
    )

    # 2. Elementwise make_real_line_rqs (dc_: Distribution Converter)
    dc_ = make_real_line_rqs(num_spline_knots2, symmetric=True, smooth=True)

    return ModuleList_([psd_block_, dc_])


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
    add("--len0", type=int)
    add("--num_spline_knots1", type=int)
    add("--num_spline_knots2", type=int)
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
