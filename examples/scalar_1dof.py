# Javad Komijani, 2021-2025

"""
This file implements inverse transform sampling using Rational Quadratic
Splines (RQS) for a quatic action The RQS parameters are trained within the
framework of normalizing flows.

To run the main function with default options, use:

    >>> python3 $filename

For parallel training, e.g., with 2 nodes and 4 processors per node, use:

    >>> torchrun --nproc_per_node=4 $filename --world_size 8

The `world_size` option serves two purposes:

1. Dividing the batch size.
2. Running `execute_ddp_training` if `world_size > 1`.
"""

import math
import torch

from normflow import Model
from normflow.nn import DistConvertor_
from normflow.action import ScalarPhi4Action
from normflow.prior import NormalPrior


# =============================================================================
def main(
    m_sq: float = -2.0,
    lambd: float = 0.2,
    lat_shape: tuple = (1,),  # 1 dof: zero dimensional lattice
    n_epochs: int = 1000,
    batch_size: int = 1024,
    knots_len: int = 10,
    load_fname: str = None,
    save_fname: str = None,
    world_size: int = 1,  # see the docstring
    print_every: int = 100,
    debug: bool = False
):
    """The main file for building and training a model using RQS."""

    if debug:
        torch.manual_seed(213)

    net_ = DistConvertor_(knots_len, symmetric=True)
    action = ScalarPhi4Action(kappa=0, m_sq=m_sq, lambd=lambd)
    prior = NormalPrior(shape=lat_shape)

    model = Model(net_=net_, prior=prior, action=action)
    model.trainer.path_gradient_autodiff = True

    if load_fname is not None:
        model.load_checkpoint(load_fname)

    train_kwargs = {
        'n_epochs': n_epochs,
        'batch_size': batch_size // world_size,
        'hyperparam': {'lr': 0.01, 'weight_decay': 0.001},
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


def _unittest(rel_tol=1e-1):
    # results vary between CPU and GPU, that's why rel_tol is so large!
    model = main(debug=True, n_epochs=50, print_every=None)
    loss = model.trainer.compute_metrics(batch_size=16)[0]
    passed = math.isclose(loss, -2.291045818475, rel_tol=rel_tol)
    if not passed:
        print(f"Unittest Failed in scalar_1dof: {loss} != -2.291045818475")
    return passed


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add = parser.add_argument

    add("--lat_shape", dest="lat_shape", type=int, nargs='+')
    add("--m_sq", dest="m_sq", type=float)
    add("--lambd", dest="lambd", type=float)
    add("--knots_len", dest="knots_len", type=int)
    add("--batch_size", dest="batch_size", type=int)
    add("--n_epochs", dest="n_epochs", type=int)
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
