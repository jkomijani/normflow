# Copyright (c) 2021-2026 Javad Komijani

"""
To run the main function with default options, use:

    >>> python $filename --config config.yaml
"""

import normflow

from normflow import Model
from normflow.prior import UniformSUnPrior
from normflow.action import MatrixAction

from normflow.nn import (
    Pade22_,
    Pade22Spline_,
    MultiChannelModule_,
    MatrixModule_
)

from normflow.lib.matrix_handles import (
    SU2MatrixParametrizer,
    SU3MatrixParametrizer
)


# =============================================================================
def main(
    n_c: int = 3,
    beta: float = 6,
    num_spline_knots: int = 2,
    n_epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 0.1,
    log_name: str = None,
    load_fname: str = None,
    save_fname: str = None
):
    """The main file for building and training the model."""

    # Define the prior distribution
    prior = UniformSUnPrior(n_c, shape=(1,))

    # Define the action for target distribution
    action = MatrixAction(beta=beta)

    # Initialize the neural network for transformations
    net_ = assemble_net(n_c, num_spline_knots)

    # Create the Model with the defined components
    model = Model(net_=net_, prior=prior, action=action)
    print("number of model parameters =", model.net_.npar)

    training_config = {
        'hyperparam': {'lr': lr},
        'log_name': log_name,
        'load_checkpoint_path': load_fname,
        'save_checkpoint_path': save_fname
    }

    model.trainer.run_training(n_epochs, batch_size, **training_config)

    normflow.reverse_flow_sanitychecker(model)

    return model


# =============================================================================
def assemble_net(n_c: int, num_spline_knots: int):
    """
    Build a network based on Pade22 (RQ) splines.

    Returns:
        Module: parameter network with:
        - n_c == 2 (SU(2)): one angle (θ) → one RQ spline
        - n_c == 3 (SU(3)): two angles (θ, φ) → two-channel RQ spline
    """
    assert 1 < n_c < 4

    # Use a simple Pade22_ if num spline knots is 2 (or less!)
    if num_spline_knots <= 2:
        param_net_ = Pade22_(n_channels=n_c - 1, channels_axis=-1)

    elif n_c == 2:
        param_net_ = Pade22Spline_(num_spline_knots)

    else:
        net0_ = Pade22Spline_(num_spline_knots)
        net1_ = Pade22Spline_((1+num_spline_knots) // 2, symmetric=True)
        param_net_ = MultiChannelModule_([net0_, net1_], channels_axis=-1)

    if n_c == 2:
        matrix_handle = SU2MatrixParametrizer()
    else:
        matrix_handle = SU3MatrixParametrizer()

    net_ = MatrixModule_(param_net_, matrix_handle=matrix_handle)
    return net_


# =============================================================================
def _unittest():
    """NOT READY YET"""


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    import yaml

    parser = ArgumentParser()
    add = parser.add_argument

    # YAML config file
    add("--config", type=str, help="Path to YAML config file")

    add("--beta", type=float)
    add("--n_c", type=str)
    add("--batch_size", type=int)
    add("--n_epochs", type=int)
    add("--num_spline_knots", type=int)
    add("--lr", type=float)
    add("--log_name", type=str)
    add("--load_fname", type=str)
    add("--save_fname", type=str)
    add("--unittest", type=bool)

    # CLI arguments

    args = vars(parser.parse_args())

    # Start with YAML config if provided
    config = {}
    if args.get("config"):
        with open(args["config"], "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # Override config with CLI args if provided
    config.update(
        {k: v for k, v in args.items() if v is not None and k != "config"}
    )

    if "unittest" in config.keys():
        _unittest()
    else:
        main(**config)
