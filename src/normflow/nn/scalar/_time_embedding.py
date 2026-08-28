# Created by Javad Komijan, 2025

"""Module for generating time-embedded weight tensors and modules."""


from typing import Dict, Tuple
import torch

from normflow.nn import RQSplineContextModule_


__all__ = ["TimeEmbeddedRQSModule_"]


# =============================================================================
class TimeEmbeddedRQSModule_(RQSplineContextModule_):
    """An RQS module whose spline parameters are conditioned on time.

    This class extends `RQSplineContextModule_` by generating spline
    parameters through a `TimeEmbeddedWeight` network. The time embedding
    produces features used to determine knot positions and/or derivatives
    in the spline.

    Args:
        n_knots (int): Number of spline knots.
        knots_x (torch.Tensor or None): Fixed x-knots, or learned if None.
        knots_y (torch.Tensor or None): Fixed y-knots, or learned if None.
        smooth (bool): If True, enforces first-derivative continuity.
        encoding_kwargs (dict): Args forwarded to `TimeEmbeddedWeight`.
        **rqs_kwargs: Additional args passed to `RQSplineContextModule_`.
    """
    def __init__(
        self,
        n_knots: int,
        knots_x: torch.Tensor | None = None,
        knots_y: torch.Tensor | None = None,
        smooth: bool = False,
        encoding_kwargs: Dict = None,
        **rqs_kwargs
    ):
        # Determine required feature dimensions
        n_x = (n_knots - 1) * (knots_x is None)
        n_y = (n_knots - 1) * (knots_y is None)
        n_d = n_knots * (not smooth)
        n_features = n_x + n_y + n_d

        # Time-conditioned feature mapping
        feature_map_fn = TimeEmbeddedWeight(
            [n_features], **(encoding_kwargs or {})
        )

        # Initialize the parent RQS module
        super().__init__(
            feature_map_fn,
            knots_x=knots_x,
            knots_y=knots_y,
            smooth=smooth,
            **rqs_kwargs
        )


# =============================================================================
class TimeEmbeddedWeight(torch.nn.Module):
    """Constructs time-emebedded weight tensors.

    Args:
        weight_shape (tuple of int): Shape of the output weight tensor,
            excluding batch dimension.
        hidden_dim (int): Hidden dimension of the MLP (default 32).
        max_freq (int): Maximum frequencey in the time encoder (default 32).
            Overlooked if `time_encoder` is provided.
        time_encoder (torch.nn.Module): Module that encodes time.
            Defaults to `SinusoidalEncoder(hidden_dim, max_freq=max_freq)`.
    """
    def __init__(
        self,
        weight_shape: Tuple[int],
        hidden_dim: int = 32,
        max_freq: int = 32,
        time_encoder: torch.nn.Module = None
    ):
        super().__init__()

        self.weight_shape = weight_shape
        n_weight = int(torch.tensor(weight_shape).prod())

        if time_encoder is None:
            time_encoder = SinusoidalEncoder(hidden_dim, max_freq=max_freq)

        self.time_encoder = time_encoder
        time_n_embed = self.time_encoder.n_embed

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(time_n_embed, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, n_weight),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time-dependent weight tensor for given time.

        Args:
            t (torch.Tensor): A tensor representing time.

        Returns:
            Tensor: Weight tensor of shape `(*t.shape, *self.weight_shape)`.
        """
        weight_t = self.mlp(self.time_encoder(t))
        return weight_t.reshape(*t.shape, *self.weight_shape)

    def set_param2zero(self):
        """Set all trainable parameters to zero."""
        for param in self.mlp.parameters():
            torch.nn.init.zeros_(param)

    def set_param2normal(self, mean: float = 0.0, std: float = 1.0):
        """Set all trainable parameters to Gaussian with given mean and std."""
        for param in self.mlp.parameters():
            torch.nn.init.normal_(param, mean=mean, std=std)


# =============================================================================
class SinusoidalEncoder(torch.nn.Module):
    """
    Implements a sinusoidal encoding inspired by "Attention Is All You Need,"
    where the frequencies change geometrically.

    Unlike the original paper where positions are integers, this class supports
    non-integer values, typically within [0, 1]. The frequency spectrum can be
    adjusted using `max_freq` and `max_freq`.

    Args:
        n_embed (int): Length of the code vector (must be even).
        min_freq (float, int): Minimum angular frequency (default is 1).
        max_freq (float, int): Maximum angular frequency (default is 1000).
        inner_ndim (int): for reshaping the output (default is 0).
        trainable_freq (bool): Frequencies are trainable (defaults to False).
        trainable_ampl (bool): Amplitudes are trainable (defaults to False).
    """

    def __init__(
        self,
        n_embed: int,
        min_freq: float = 1.,
        max_freq: float = 1000.,
        inner_ndim: int = 0,
        trainable_freq: bool = False,
        trainable_ampl: bool = False
    ):

        assert n_embed % 2 == 0, "Embedding length must be even."

        super().__init__()

        self.n_embed = n_embed
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.inner_ndim = inner_ndim
        self.trainable_freq = trainable_freq
        self.trainable_ampl = trainable_ampl

        if trainable_freq:
            self.freq_ratio = torch.nn.Parameter(torch.rand(n_embed // 2))
        else:
            power = torch.arange(n_embed // 2) * (2 / n_embed)  # \in [0, 1)
            freq = max_freq / (max_freq / min_freq)**power
            self.register_buffer('freq', freq)

        if trainable_ampl:
            self.ampl = torch.nn.Parameter(torch.randn(n_embed))
        else:
            self.ampl = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the sinusoidal encoding of t.

        Args:
            t (torch.Tensor): The input tensor, e.g., representing time.

        Returns:
            torch.Tensor: A tensor of original shape `(*t.shape, n_embed)` with
                sinusoidal encoding. It is then reshaped to have `inner_ndim`
                additional inner dimensions with unit lenght.
        """
        if self.trainable_freq:
            angle = t.unsqueeze(-1) * (self.freq_ratio * self.max_freq)
        else:
            angle = t.unsqueeze(-1) * self.freq

        encoded_t = torch.zeros((*t.shape, self.n_embed), device=t.device)

        encoded_t[..., 0::2] = torch.sin(angle)
        encoded_t[..., 1::2] = torch.cos(angle)

        if self.trainable_ampl:
            encoded_t = self.ampl * encoded_t

        out_shape = (*t.shape, self.n_embed, *(1,) * self.inner_ndim)
        return encoded_t.reshape(*out_shape)
