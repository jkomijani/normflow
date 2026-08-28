# Javad Komijani, 2022-2026

"""
To run the main function with default options, use:

    >>> python $filename --config config.yaml

For parallel training, e.g., with (2 nodes) and 4 processors per node, use:

    >>> torchrun --nproc_per_node=4 $filename --config config.yaml
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from functools import partial

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import normflow

from normflow import Model

from normflow.prior import UniformU1Prior, UniformSUnPrior

from normflow.action import WilsonU1GaugeAction, WilsonGaugeAction

from normflow.nn import (
    GaugeSLinkModule_,
    GaugeSingularValueModule_,
    GaugeModuleList_,
    SpectralStateTransform_,
    ModuleList_,
    Pade22_,
    Pade22DualCoupling_,
    RQSplineNet_,
    MultiChannelModule_,
    InvisibilityMaskWrapperModule_,
    ModalMatrixSteppedCommutatorFlow_,
    DenseBlock
)

from normflow.lib.matrix_handles import (
    U1WilsonStaplesHandle,
    WilsonStaplesHandle,
    U1Parametrizer,
    SU2MatrixParametrizer,
    SU3MatrixParametrizer
)

from normflow.mask import (
    EvenOddMask,
    FourWayParityMask
)

# torch.set_default_dtype(torch.float64)


# =============================================================================
def main(
    beta: float = 1,
    gauge: str = 'SU(3)',
    lat_shape: tuple = (4, 4, 4, 4),
    n_epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    path_gradient_autodiff: bool = True,
    alpha_tmax: int | None = None,
    log_name: str = None,
    load_fname: str = None,
    save_fname: str = None,
    debug: bool = False,
    **net_kwargs
):
    """The main file for building and training the model."""

    if debug:
        torch.manual_seed(42)

    links_shape = (*tuple(lat_shape), len(lat_shape))

    n_c = int(gauge[-2])  # number of colors of the guage group
    if n_c == 1:
        action = WilsonU1GaugeAction(beta=beta)
        prior = UniformU1Prior(shape=links_shape)
    else:
        action = WilsonGaugeAction(beta=beta)
        prior = UniformSUnPrior(
            n=n_c, shape=links_shape, drop_constant_log_prob=True
        )

    net_ = assemble_net(
        lat_shape=lat_shape, n_c=n_c, action=action, **net_kwargs
    )

    model = Model(network_fn_=net_, prior=prior, action=action)

    training_config = {
        'hyperparam': {'lr': lr, 'weight_decay': weight_decay},
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
    lat_shape: tuple,
    n_c: int,
    n_layers: int = 1,
    num_spline_knots: int = 5,
    add_dual_param_net: bool = False,
    add_eigvecs_net: bool = False,
    dual_net_hidden_sizes: tuple = (8, 8),
    add_triv_map: bool = False,
    action=None  # needed only if add_triv_map == True
):
    """Assemble the net for the gauge theory."""

    mask_shape = (*lat_shape, 1)  # 1 for the eigvals axis

    eigvecs_mask_shape = (*lat_shape, 1, 1)  # 1, 1 for the matrix axes

    matrix_handle, staples_handle = get_handles(n_c)

    def make_gauge_slink_module_(mu, parity, suppress_flag=False):
        """Build and return an instance of GaugeSLinkModule_"""
        mask = EvenOddMask(shape=mask_shape, parity=parity, exclude_mu=mu)
        param_net_ = build_param_net(n_c, mask, num_spline_knots)

        if add_dual_param_net and not suppress_flag:
            dual_param_net_ = build_dual_param_net(
                n_c, mask, dual_net_hidden_sizes
            )
        else:
            dual_param_net_ = None

        if add_eigvecs_net and not suppress_flag:
            eigvecs_mask = EvenOddMask(
                shape=eigvecs_mask_shape, parity=parity, exclude_mu=mu
            )
            eigvecs_net_ = build_eigvecs_net(n_c, eigvecs_mask)
            extra_ops = [eigvecs_net_]
        else:
            extra_ops = None

        slink_transform_ = SpectralStateTransform_(
            matrix_handle,
            param_net_,
            dual_param_net_=dual_param_net_,
            extra_ops=extra_ops
        )
        gauge_module_ = GaugeSLinkModule_(
            mu=mu,
            nu_list=[nu for nu in range(ndim) if nu != mu],
            staples_handle=staples_handle,
            slink_transform_=slink_transform_
        )
        return gauge_module_

    def make_gauge_q_transform_module_(mu, mask_id):
        """Build and return an instance of GaugeSingularValueModule_"""
        mask = FourWayParityMask(
            shape=mask_shape, mask_id=mask_id, zebra_mu=mu
        )
        param_net_ = build_param_net(n_c, mask, num_spline_knots)

        q_transform_ = SpectralStateTransform_(matrix_handle, param_net_)

        gauge_module_ = GaugeSingularValueModule_(
            mu=mu,
            nu_list=[nu % ndim for nu in range(mu + 1, ndim + mu)],
            staples_handle=staples_handle,
            q_transform_=q_transform_,
            staples_kwargs={'staples_coeff': '010000'}
        )
        return gauge_module_

    ndim = len(lat_shape)
    nets_ = []

    if ndim == 4:
        mu_list = [0, 2, 1, 3]
    else:
        mu_list = range(ndim)

    for n in range(n_layers):
        for mu in mu_list:
            for parity in range(2):
                mask_id = (2 * n + 2 * mu - parity) % 4
                # suppress_flag = n < (n_layers - 1)
                nets_.append(make_gauge_q_transform_module_(mu, mask_id))
                nets_.append(make_gauge_slink_module_(mu, parity))

    net_ = GaugeModuleList_(nets_)

    if add_triv_map:
        triv_map_ = normflow.nn.WilsonTrivMap_(action)
        net_ = ModuleList_([triv_map_, net_])

    return net_


# =============================================================================
def build_param_net(n_c, mask, num_spline_knots):
    """
    Build a masked parameter network based on Pade22 (RQ) splines.

    All networks are wrapped with `InvisibilityMaskWrapperModule_`.

    Args:
        n_c (int): Number of colors.
        mask: Visibility mask for the wrapper module.
        num_spline_knots (int):  Number of knots in the Pade22 splines.

    Returns:
        Module: Masked parameter network with:
        - n_c == 1 (U(1)): one angle → one RQ spline
        - n_c == 2 (SU(2)): one angle (θ) → one RQ spline
        - n_c == 3 (SU(3)): two angles (θ, φ) → two-channel RQ spline
    """
    # Use a simple Pade22_ if num spline knots is 2 (or less!)
    if num_spline_knots <= 2:
        n_channels = max(1, n_c - 1)  # channels ~ independent angles
        net_ = Pade22_(n_channels=n_channels, channels_axis=-1)
        return InvisibilityMaskWrapperModule_(net_, mask=mask)

    # n_c == 1 & 2, i.e. U(1) & SU(2): a single θ
    if n_c < 3:
        net_ = RQSplineNet_(num_spline_knots)
        return InvisibilityMaskWrapperModule_(net_, mask=mask)

    # n_c == 3, i.e. SU(3): two angles (θ, φ) → two spline channels
    if n_c == 3:
        par0_net_ = InvisibilityMaskWrapperModule_(
            RQSplineNet_(num_spline_knots), mask=mask
        )
        par1_net_ = InvisibilityMaskWrapperModule_(
            RQSplineNet_((1+num_spline_knots) // 2, symmetric=True), mask=mask
        )
        # combine channels and return
        return MultiChannelModule_([par0_net_, par1_net_], channels_axis=-1)

    return None  # not implemented


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

    assert n_c > 1

    # in_features:
    #     if `n_c > 2`: n_c singular values times phases -> `2 n_c` real values
    #     if `n_c == 2`: 2 singular values that are equal -> 1 real value
    in_features = 2 * n_c if n_c > 2 else 1
    out_features = 2 * (n_c - 1)  # times 2 because Pade22 has 2 params

    dense_dict = {
        'in_features': in_features,
        'out_features': out_features,
        'hidden_sizes': hidden_sizes,
        'acts': acts,
        'features_axis': -1
    }

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
        ModalMatrixSteppedCommutatorFlow_ or None: A network to learn modal
            matrix transformations, or None if n_c < 3.
    """
    if n_c < 3:
        return None

    # tau_par = torch.nn.Parameter(-torch.rand((1,)))
    tau_par = torch.nn.Parameter(torch.zeros((1,)))
    net_ = ModalMatrixSteppedCommutatorFlow_(tau_par=tau_par, mask=mask)
    net_.flow_.n_steps = 1
    net_.flow_.reverse_mode_iter = 4
    return net_


# =============================================================================
def get_handles(n_c):
    """
    Returns the appropriate matrix and staples handlers based on the number of
    colors.

    These handlers are used in lattice gauge theory models to update link
    variables using staples, with parametrizations that vary depending on the
    gauge group (U(1), SU(2), SU(3)).

    Args:
        n_c (int): Number of colors of the gauge group.

    Returns:
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

    return matrix_handle, staples_handle


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    import yaml

    parser = ArgumentParser()
    add = parser.add_argument

    # YAML config file
    add("--config", type=str, help="Path to YAML config file")

    add("--lat_shape", type=int, nargs='+')
    add("--beta", type=float)
    add("--gauge", type=str)
    add("--batch_size", type=int)
    add("--n_epochs", type=int)
    add("--n_layers", type=int)
    add("--num_spline_knots", type=int)
    add("--add_triv_map", type=bool)
    add("--path_gradient_autodiff", type=bool)
    add("--lr", type=float)
    add("--weight_decay", type=float)
    add("--add_dual_param_net", type=bool)
    add("--add_eigvecs_net", type=bool)
    add("--alpha_tmax", type=int)
    add("--log_name", type=str)
    add("--load_fname", type=str)
    add("--save_fname", type=str)

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

    main(**config)
