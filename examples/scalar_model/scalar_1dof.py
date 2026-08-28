# Javad Komijani, 2021-2026

"""
This file implements inverse transform sampling using Rational Quadratic
Splines (RQS) for a quartic action. The RQS parameters are trained within
the framework of normalizing flows.
"""

import torch

from normflow import Model
from normflow.nn import make_real_line_rqs
from normflow.action import ScalarPhi4Action
from normflow.prior import NormalPrior


# =============================================================================
def main(
    m_sq: float = -2.0,
    lambd: float = 0.2,
    lat_shape: tuple = (1,),  # 1 dof: zero dimensional lattice
    n_epochs: int = 1000,
    batch_size: int = 1024,
    num_spline_knots: int = 10,
    load_fname: str = None,
    save_fname: str = None,
    debug: bool = False
):
    """
    Build, configure, and train a 1-dof RQS model.

    Steps:
        1. Build the network (`make_real_line_rqs`), action, and prior.
        2. Wrap into a `Model`.
        3. Run training via `model.trainer.run_training`.

    Args:
        m_sq, lambd: Scalar action parameters.
        lat_shape: Shape of the lattice (single site for 1 dof).
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        num_spline_knots: Number of RQS knots in `make_real_line_rqs`.
        load_fname: Path to load a checkpoint before training.
        save_fname: Path to save a checkpoint after training.
        debug: If True, sets a fixed random seed for reproducibility.

    Returns:
        Model: The trained model instance.
    """

    if debug:
        torch.manual_seed(213)

    net_ = make_real_line_rqs(num_spline_knots, symmetric=True)
    action = ScalarPhi4Action(kappa=0, m_sq=m_sq, lambd=lambd)
    prior = NormalPrior(shape=lat_shape)

    model = Model(net_=net_, prior=prior, action=action)

    training_config = {
        'hyperparam': {'lr': 0.01, 'weight_decay': 0.001},
        'path_gradient_autodiff': True,
        'load_checkpoint_path': load_fname,
        'save_checkpoint_path': save_fname
    }

    model.trainer.run_training(n_epochs, batch_size, **training_config)

    ess = model.compute_metrics(batch_size=batch_size)[0]
    print("ESS", ess)

    return model


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add = parser.add_argument

    add("--lat_shape", type=int, nargs='+')
    add("--m_sq", type=float)
    add("--lambd", type=float)
    add("--num_spline_knots", type=int)
    add("--n_epochs", type=int)
    add("--batch_size", type=int)
    add("--load_fname", type=str)
    add("--save_fname", type=str)

    args = vars(parser.parse_args())
    args = {key: value for key, value in args.items() if value is not None}

    main(**args)
