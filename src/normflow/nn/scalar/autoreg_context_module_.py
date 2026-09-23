# Copyright (c) 2026 Javad Komijani

"""
This module contains an autoregressive module in the masked parameterization:
one feature map, evaluated on the whole input, whose output is masked so that
the parameters of component `i` depend only on the components before it.

The transformation classes defined here are subclasses of `Module_`, and like
it, the trailing underscore implies that the associated forward and reverse
methods handle the Jacobians of the transformation.

Example
-------
    >>> net_ = make_rqs_masked_autoreg_context_module(
    ...     data_features=4, context_features=3, n_segments=8
    ... )
    >>> x = torch.rand(32, 4)              # latent,  (batch, data_features)
    >>> c = torch.rand(32, 3)              # context, (batch, context_features)
    >>> y, logj = net_.forward(x, args=c)  # latent -> data,  (32, 4)

Omit `context_features` for an unconditional flow; `args` is then unused.

`make_rqs_masked_autoreg_context_module` is a convenience constructor and
doubles as an example of how to build a `MaskedAutoRegContextModule_`;
it may change.
"""

# pylint: disable=invalid-name, relative-beyond-top-level
# pylint: disable=too-many-arguments, too-many-positional-arguments
# pylint: disable=too-many-locals, arguments-renamed
# pylint: disable=too-few-public-methods

from typing import List, Sequence

import torch

from .._core import Module_

from .modules import DenseBlock
from .rqs_modules_ import RQSplineContextModule_


__all__ = [
    "MaskedAutoRegContextModule_",
    "make_rqs_masked_autoreg_context_module",
]


Tensor = torch.Tensor  # for typing


# =============================================================================
class MaskedAutoRegContextModule_(Module_):
    """Masked autoregressive module, `y_i = T(x_i ; f(y)_i, *context)`.

    Combines the frozen parts with the external context and hands them to
    `context_module_` as `args`. Its feature map must be masked, returning
    one parameter set per component; the input gains a trailing singleton
    axis to match.

    Conditioning is on the data side (MAF), so `reverse` takes a single pass,
    while `forward` iterates once per component, each fixing one more, so the
    result is exact rather than approximate. Swapping the two gives IAF.
    """

    def __init__(self, context_module_: Module_):
        super().__init__()
        self.context_module_ = context_module_

    def forward(self, x, log0=0, args=None):
        """Transform latent variables into data, one component per pass."""
        y = x  # seeded with `x`; 0 may be outside the domain
        for _ in range(x.shape[-1]):
            y, logj = self.context_module_.forward(
                x.unsqueeze(-1), args=(y, *self._as_args_tuple(args))
            )
            y = y.squeeze(-1)
        # Remark: only logj computed in the last step of loop matters
        return y, log0 + logj

    def reverse(self, x, log0=0, args=None):
        """Transform data into latent variables in a single pass."""
        y, logj = self.context_module_.reverse(
            x.unsqueeze(-1), args=(x, *self._as_args_tuple(args))
        )
        return y.squeeze(-1), log0 + logj

    @staticmethod
    def _as_args_tuple(args):
        """Normalize `args` into a tuple, as `ContextModule_` does."""
        if isinstance(args, tuple):
            return args
        return () if args is None else (args,)


# =============================================================================
class ConcatInputs(torch.nn.Module):
    """Concatenate inputs along the last dimension before applying `net`."""

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, *xs: Tensor) -> Tensor:
        """Concatenate, then apply `net`."""
        return self.net(torch.cat(xs, dim=-1))


# =============================================================================
def make_autoreg_weight_masks(
    in_features: int,
    out_features_per_var: int,
    hidden_sizes: Sequence[int],
    context_features: int = None
) -> List[Tensor]:
    """
    Create per-layer weight masks for an autoregressive network.

    The last `context_features` inputs are unrestricted context, while the
    remaining inputs are autoregressive variables. Each variable has
    `out_features_per_var` outputs.

    Connections satisfy `degree_out >= degree_in` in hidden layers and
    `degree_out > degree_in` in the output layer, preventing a variable from
    depending on itself.

    Returns one `(n_out, n_in)` mask per layer.

    Example:
        >>> make_autoreg_weight_masks(3, 1, (), context_features=1)
        [tensor([[0., 0., 1.],
                 [1., 0., 1.]])]
        # var1 sees only context, var2 sees var1 & context, neither sees itself
    """
    context_features = context_features or 0
    data_features = in_features - context_features
    assert data_features > 0, "context_features must be less than in_features"

    total_out_features = out_features_per_var * data_features
    sizes = (in_features, *hidden_sizes, total_out_features)

    context_degrees = torch.zeros(context_features, dtype=torch.long)
    degrees = [
        torch.cat([torch.arange(1, data_features + 1), context_degrees])
    ]

    # Degree 0 provides a context-only path to the first output.
    # The maximum hidden degree is data_features - 1, so the strict
    # output mask always has a valid connection.
    for width in hidden_sizes:
        degrees.append(torch.arange(width) % data_features)

    degrees.append(
        1 + torch.arange(data_features).repeat_interleave(out_features_per_var)
    )

    masks = []
    for ind in range(len(sizes) - 1):
        d_in, d_out = degrees[ind], degrees[ind + 1]

        # Use a strict inequality at the output to prevent self-dependence.
        last = ind == len(sizes) - 2
        mask = (d_out[:, None] > d_in[None, :]) if last else \
               (d_out[:, None] >= d_in[None, :])
        masks.append(mask.to(torch.get_default_dtype()))
    return masks


# =============================================================================
def make_rqs_masked_autoreg_context_module(
    data_features: int,
    n_segments: int = 8,
    hidden_sizes: Sequence[int] = (32, 32),
    context_features: int = None,
    smooth: bool = False,
    **kwargs
) -> MaskedAutoRegContextModule_:
    """
    Build a masked autoregressive rational-quadratic-spline flow.
    """
    in_features = data_features + (context_features or 0)
    out_features_per_var = (2 * n_segments) if smooth else (3 * n_segments + 1)
    total_out_features = out_features_per_var * data_features

    masks = make_autoreg_weight_masks(
        in_features,
        out_features_per_var,
        hidden_sizes=hidden_sizes,
        context_features=context_features
    )

    dense_block = DenseBlock(
        in_features,
        total_out_features,
        hidden_sizes,
        acts=(*[torch.nn.SiLU() for _ in hidden_sizes], None),
        masks=masks
    )
    # zero output weights => zero features => identity transformation.
    # Only the output layer: zeroing all of them would leave every hidden
    # activation at zero, hence no gradient anywhere but the output bias.
    dense_block.set_param2zero(n_layer=-1)

    # the variables come first and the context last, matching the order the
    # degrees above assume
    feature_map_fn = ConcatInputs(torch.nn.Sequential(
        dense_block,
        torch.nn.Unflatten(-1, (data_features, out_features_per_var))
    ))
    return MaskedAutoRegContextModule_(
        RQSplineContextModule_(feature_map_fn, smooth=smooth, **kwargs)
    )
