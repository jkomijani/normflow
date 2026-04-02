# Copyright (c) 2026 Javad Komijani

"""
An invertible gauge-equivariant UNet.
"""

# pylint: disable=invalid-name, arguments-renamed

from typing import List, Tuple
import torch

from ._core import Module_


__all__ = ["UNet_"]


# =============================================================================
class UNet_(Module_):
    """
    Invertible U-Net architecture with log-Jacobian tracking.

    This module composes invertible transformations in a U-Net structure:

        Forward flow:  x → encoder → bottleneck → decoder → y
        Reverse flow:  y → decoder⁻¹ → bottleneck⁻¹ → encoder⁻¹ → x

    Each component is itself composed of ``Module_`` instances, allowing
    exact inversion and cumulative log-Jacobian computation across the
    entire architecture.

    Args:
        encoder_layers (List[Module_]):
            Sequence of invertible layers applied in the encoder.

        bottleneck (Module_):
            Invertible transformation applied at the lowest resolution.

        decoder_layers (List[Module_]):
            Sequence of invertible layers applied in the decoder.

        downsampler (torch.nn.Module):
            Invertible module responsible for downsampling (encoder) and
            upsampling (decoder). Must implement:
                - ``forward(data) -> (data_down, skip)``
                - ``reverse(data_down, skip) -> data``

    Shape constraints:
        The user must ensure that all components are shape-compatible,
        especially with respect to skip connections.
    """

    def __init__(
        self,
        encoder_layers: List[Module_],
        bottleneck_: Module_,
        decoder_layers: List[Module_],
        downsampler: torch.nn.Module
    ):
        super().__init__()
        self.encoder_ = UNetEncoder_(encoder_layers, downsampler)
        self.bottleneck_ = bottleneck_
        self.decoder_ = UNetDecoder_(decoder_layers, downsampler)

    def forward(self, data: torch.Tensor, log0=0):
        """
        Apply the forward (encoder → bottleneck → decoder) transformation.

        Args:
            data (Tensor): Input tensor.
            log0 (Tensor | float): Initial log-Jacobian value. Defaults to 0.

        Returns:
            Tuple[Tensor, Tensor]:
                - Transformed tensor
                - Accumulated log-Jacobian
        """
        (data, skips), logj = self.encoder_(data, log0=log0)
        data, logj = self.bottleneck_(data, log0=logj)
        data, logj = self.decoder_((data, skips), log0=logj)
        return data, logj

    def reverse(self, data: torch.Tensor, log0=0):
        """
        Apply the inverse (decoder → bottleneck → encoder) transformation.

        Args:
            data (Tensor): Input tensor in output space.
            log0 (Tensor | float): Initial log-Jacobian value. Defaults to 0.

        Returns:
            Tuple[Tensor, Tensor]:
                - Reconstructed input tensor
                - Accumulated log-Jacobian
        """
        (data, skips), logj = self.decoder_.reverse(data, log0=log0)
        data, logj = self.bottleneck_.reverse(data, log0=logj)
        data, logj = self.encoder_.reverse((data, skips), log0=logj)
        return data, logj


class UNetEncoder_(Module_):
    """
    Invertible encoder block used in a U-Net architecture.

    This module applies a sequence of invertible layers interleaved with
    invertible downsampling operations, while storing intermediate outputs
    ("skips") for use in the decoder.

    Args:
        layers (List[Module_]): Sequence of invertible transformations.
        downsampler (torch.nn.Module): Invertible downsampling module providing
            - ``forward(data) -> (data_down, skip)``
            - ``reverse(data_down, skip) -> data``
    """
    def __init__(self, layers, downsampler):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.downsampler = downsampler

    def forward(self, data: torch.Tensor, log0=0):
        """
        Apply encoder transformations and collect skip connections.

        Args:
            data (Tensor): Input tensor.
            log0 (Tensor | float): Initial log-Jacobian value. Defaults to 0.

        Returns:
            Tuple[(Tensor, List[Tensor]), Tensor]:
                - (encoded tensor, skip tensors)
                - accumulated log-Jacobian
        """
        skips = []
        logj = log0

        for layer_ in self.layers:
            data, logj = layer_(data, log0=logj)
            data_down, skip = self.downsampler(data)
            skips.append(skip)
            data = data_down  # for next round

        return (data, skips), logj

    def reverse(self, data_and_skips: Tuple[torch.Tensor, list], log0=0):
        """
        Invert the encoder transformations.

        Args:
            data_and_skips (Tuple[Tensor, List[Tensor]]):
                Encoded tensor and corresponding skip tensors.
            log0 (Tensor | float): Initial log-Jacobian value. Defaults to 0.

        Returns:
            Tuple[Tensor, Tensor]:
                - Reconstructed tensor
                - accumulated log-Jacobian
        """
        data_down, skips = data_and_skips
        logj = log0

        assert len(self.layers) == len(skips), "Mismatch in number of skips"

        for layer_, skip in zip(reversed(self.layers), reversed(skips)):
            data = self.downsampler.reverse(data_down, skip)
            data, logj = layer_.reverse(data, log0=logj)
            data_down = data  # for next round

        return data, logj


class UNetDecoder_(UNetEncoder_):
    """
    Invertible decoder block for a U-Net architecture.

    This module mirrors ``UNetEncoder_`` and reuses its logic by swapping
    forward and reverse operations.

    Conceptually:
        decoder.forward  ≡ encoder.reverse
        decoder.reverse  ≡ encoder.forward
    """

    def forward(self, data_and_skips: Tuple[torch.Tensor, list], log0=0):
        """
        Apply decoder (inverse encoder) transformation.
        """
        return super().reverse(data_and_skips, log0=log0)

    def reverse(self, data: torch.Tensor, log0=0):
        """
        Apply inverse decoder (forward encoder) transformation.
        """
        return super().forward(data, log0=log0)
