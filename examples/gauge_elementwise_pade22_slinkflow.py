# Javad Komijani, 2022-2025

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

from functools import partial

import torch
import normflow

from normflow import Model

from normflow.prior import (
    U1Prior,
    SUnPrior
)

from normflow.action import (
    U1GaugeAction,
    GaugeAction
)

from normflow.nn import (
    GaugeModule_,
    GaugeModuleList_,
    ModuleList_,
    Pade22_,
    Pade22DualCoupling_,
    InvisibilityMaskWrapperModule_,
    ModalMatrixFlow_,
    DenseBlock
)

from normflow.lib.matrix_handles import (
    U1WilsonStaplesHandle,
    WilsonStaplesHandle,
    U1Parametrizer,
    SU2MatrixParametrizer,
    SU3MatrixParametrizer
)

from normflow.mask import EvenOddMask


# =============================================================================
def main(
    # Lattice setup
    beta: float = 1,
    gauge: str = 'SU(3)',
    lat_shape: tuple = (4, 4, 4, 4),
    # Training setup
    n_epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 0.01,
    path_gradient_autodiff: bool = True,
    alpha_tmax: bool = None,
    world_size: int = 1,
    print_every: int = 10,
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

    n_c = int(gauge[-2])  # number of colors of the guage group

    links_shape = (len(lat_shape), *tuple(lat_shape))

    if n_c == 1:
        action = U1GaugeAction(beta=beta, ndim=len(lat_shape))
        prior = U1Prior(shape=links_shape)
    else:
        action = GaugeAction(beta=beta, ndim=len(lat_shape), n_c=n_c)
        prior = SUnPrior(n=n_c, shape=links_shape, drop_constant_log_prob=True)

    net_ = assemble_net(
        lat_shape=lat_shape, n_c=n_c, action=action, **net_kwargs
    )

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
        'hyperparam': {'lr': lr, 'weight_decay': 1e-3, 'betas': (0.9, 0.99)},
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
    lat_shape: tuple,
    n_c: int,
    n_layers: int = 1,
    add_eigvecs_net: bool = True,
    add_eigangs_net: bool = False,
    add_dual_param_net: bool = True,
    dual_net_hidden_sizes: tuple = (8, 8),
    inner_net_hidden_sizes: tuple = (2, 4, 4),
    add_triv_map: bool = False,
    action=None  # needed only if add_triv_map == True
):
    """Assemble the net for the gauge theory."""

    mask_shape = (*lat_shape, 1)  # 1 for the eigvals axis
    masks_list = [EvenOddMask(shape=mask_shape, parity=p) for p in range(2)]

    shape = (*lat_shape, 1, 1)  # 1, 1 for the matrix axes
    eigvecs_masks_list = [EvenOddMask(shape=shape, parity=p) for p in range(2)]

    handles_dict = set_staples_handles(n_c)
    ndim = len(lat_shape)
    nets_ = []

    # The folowing ones will be updated in the loop
    dual_param_net_ = None
    eigvecs_net_ = None
    eigangs_net_ = None

    for _ in range(n_layers):

        for parity in range(2):

            mask = masks_list[parity]
            dual_param_mask = mask
            eigvecs_mask = eigvecs_masks_list[parity]

            for mu in range(ndim):
                nu_list = [nu for nu in range(ndim) if nu != mu]
                param_net_ = build_param_net(n_c, mask)

                if add_dual_param_net:
                    dual_param_net_ = build_dual_param_net(
                        n_c, dual_param_mask, dual_net_hidden_sizes
                    )

                if add_eigvecs_net:
                    eigvecs_net_ = build_eigvecs_net(n_c, eigvecs_mask)

                if add_eigangs_net:
                    eigangs_net_ = build_eigangs_net(n_c, mask)

                elementwise_net_ = GaugeModule_(
                    mu=mu,
                    nu_list=nu_list,
                    param_net_=param_net_,
                    dual_param_net_=dual_param_net_,
                    eigvecs_net_=eigvecs_net_,
                    eigangs_net_=eigangs_net_,
                    **handles_dict
                )
                nets_.append(elementwise_net_)

    net_ = GaugeModuleList_(nets_)

    if add_triv_map:
        triv_map_ = normflow.nn.WilsonTrivMap_(action)
        net_ = ModuleList_([triv_map_, net_])

    return net_


# =============================================================================
def build_param_net(n_c, mask):
    """
    Constructs a parameter network wrapped with an invisibility mask.

    This network is based on a Pade22_ architecture with adjusted channel size.

    Args:
        n_c (int): Number of colors of the gauge group.
                   (If less than 2, defaults to 1.)
        mask: A mask used to control visibility in the wrapper module.

    Returns:
        A wrapped Pade22_ network with the invisibility mask applied.
    """
    net_ = Pade22_(n_channels=max(1, n_c - 1), channels_axis=-1)
    return InvisibilityMaskWrapperModule_(net_, mask=mask)


def build_dual_param_net(n_c, mask, hidden_sizes):
    """
    Constructs an element-wise neural network using dual parameters for
    parameter updates.

    The network internally uses a DenseBlock module to predict parameters for
    a Pade22DualCoupling_ block.

    Args:
        n_c (int): Number of colors of the gauge group.
        mask: A mask for controlling component-wise operations.
        hidden_sizes (typle): Sizes of the hidden layers in the MLP.

    Returns:
        Pade22DualCoupling_: A dual-parameter coupling network configured with
                             the given structure.
    """

    # acts = (*[torch.nn.LeakyReLU()]*len(hidden_sizes), None)
    acts = (*[torch.nn.SiLU()]*len(hidden_sizes), None)

    assert n_c in (2, 3)

    # in_features:
    #     if n_c == 3: 3 singular values + cos/sin of phase
    #     if n_c == 2: 1 (or 2 ?)
    in_features = 5 if n_c == 3 else 1
    out_features = 2 * (n_c - 1)  # times 2 because Pade22 has 2 params

    dense_dict = dict(
        in_features=in_features,
        out_features=out_features,
        hidden_sizes=hidden_sizes,
        acts=acts,
        features_axis=-1
    )

    net_ = Pade22DualCoupling_(
        [DenseBlock(**dense_dict)], mask=mask, channels_axis=-1
    )

    # Initialize parameters with small standard deviation
    for net in net_.nets:
        net.set_param2normal(std=0.01)

    return net_


def build_eigvecs_net(n_c, mask):
    """
    Constructs a network to learn eigenvector transformations for a normalizing
    flow.

    This is only applicable for networks with at least 3 channels.

    Args:
        n_c (int): Number of colors of the gauge group.
                   Must be >= 3 to construct the network.
        mask: Mask applied to the flow for controlling component interaction.

    Returns:
        ModalMatrixFlow_ or None: A network to learn modal matrix
                                  transformations, or None if n_c < 3.
    """
    if n_c < 3:
        return None
    else:
        tau_par = torch.nn.Parameter(torch.tensor([-1.]))
        net_ = ModalMatrixFlow_(tau_par=tau_par, mask=mask)
        net_.flow_.n_steps = 4
        net_.flow_.reverse_mode_iter = 4
        return net_


def build_eigangs_net(n_c, mask):
    """
    Constructs a spectral flow network for learning eigen-angles (phases) of
    unitary matrices.

    The implementation depends on the number of colors `n_c`:
      - For SU(2): uses SU2SpectralFlow_
      - For SU(n > 2): uses SUnSpectralFlow_

    Args:
        n_c (int): Number of colors of the gauge group. (Must be >= 2.)
        mask: Mask applied to control element-wise flow behavior.

    Returns:
        A spectral flow module for eigenvalue phase transformation.
    """
    source = normflow.nn.gauge.unitary_flow_
    if n_c == 2:
        tau_par = torch.nn.Parameter(torch.tensor([0.]))
        net_ = source.SU2SpectralFlow_(tau_par=tau_par, mask=mask)
    else:
        tau_par = torch.nn.Parameter(torch.tensor([0.1]))
        net_ = source.SUnSpectralFlow_(n_c, tau_par=tau_par, mask=mask)
    return net_


# =============================================================================
def set_staples_handles(n_c):
    """
    Returns the appropriate matrix and staples handlers based on the number of
    colors.

    These handlers are used in lattice gauge theory models to update link
    variables using staples, with parametrizations that vary depending on the
    gauge group (U(1), SU(2), SU(3)).

    Args:
        n_c (int): Number of colors of the gauge group.

    Returns:
        dict: A dictionary with:
            - 'matrix_handle': Parametrizer for matrix representation (n_c < 4)
            - 'staples_handle': Corresponding staples handler for gauge updates
    """

    if n_c == 1:
        staples_handle = U1WilsonStaplesHandle()
    else:
        staples_handle = WilsonStaplesHandle()

    if n_c == 1:
        matrix_handle = U1Parametrizer()
    elif n_c == 2:
        matrix_handle = SU2MatrixParametrizer()
    elif n_c == 3:
        matrix_handle = SU3MatrixParametrizer()
    else:
        raise ValueError(f"Unsupported n_c={n_c}; only 1, 2, 3 are supported.")

    return {'matrix_handle': matrix_handle, 'staples_handle': staples_handle}


# =============================================================================
def _unittest():
    print("""NOT IMPLEMENTED YET""")


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add = parser.add_argument

    # Lattice setup
    add("--lat_shape", dest="lat_shape", type=int, nargs='+')
    add("--beta", dest="beta", type=float)
    add("--gauge", dest="gauge", type=str)
    # Architecture setup
    add("--n_layers", dest="n_layers", type=int)
    add("--add_triv_map", dest="add_triv_map", type=bool)
    add("--add_eigvecs_net", dest="add_eigvecs_net", type=bool)
    add("--add_eigangs_net", dest="add_eigangs_net", type=bool)
    add("--add_dual_param_net", dest="add_dual_param_net", type=bool)
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
