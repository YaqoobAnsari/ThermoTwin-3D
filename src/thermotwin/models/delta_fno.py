"""Delta-FNO — predict a *correction* on top of the analytic 1-D theta prior.

U-value is set by the through-wall temperature gradient at the indoor face, which
the closed-form 1-D conduction solution already nails for a clear (bridge-free)
wall. The enriched ``feature_set`` carries that analytic field as a dedicated input
channel (``clearwall_index``). Instead of asking the FNO to regress the whole field
from scratch — where spectral bias smears the sharp bridge gradients that drive
U-MAE — we make it learn only the residual ``theta - theta_prior``. The prior
handles the smooth bulk; the network spends all its capacity on the localized
bridge corrections.

The forward is ``theta_prior + fno(x)`` where ``theta_prior`` is read straight from
the input tensor, so the prior costs nothing and the contract stays the strict
``(B, C, H, W) -> (B, 1, H, W)``. Requires the ``enriched`` feature set (>= 4
channels), since it relies on the clearwall channel being present.
"""

from __future__ import annotations

from neuralop.models import FNO
from torch import Tensor, nn

__all__ = ["DeltaFNO", "build_delta_fno"]


class DeltaFNO(nn.Module):
    """FNO that predicts a correction added to the analytic 1-D theta prior.

    Args:
        in_channels: input field channels (default 6 = enriched feature set).
        out_channels: output channels (1 = theta).
        n_modes: retained Fourier modes per axis (through-wall, along-wall).
        hidden_channels: width of the spectral-convolution channels.
        n_layers: number of Fourier layers.
        clearwall_index: channel index holding the analytic 1-D clear-wall theta
            field used as the additive prior.
        domain_padding: spectral-conv domain padding fraction, passed to ``FNO``.
            ``[0.25, 0.0]`` pads only the non-periodic through-wall axis.
        positional_embedding: coordinate embedding for the inner FNO.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        n_modes: tuple[int, int] = (8, 16),
        hidden_channels: int = 32,
        n_layers: int = 4,
        clearwall_index: int = 3,
        domain_padding: list[float] | None = None,
        positional_embedding: str | nn.Module = "grid",
    ) -> None:
        super().__init__()
        if clearwall_index < 0 or clearwall_index >= in_channels:
            raise ValueError(
                f"clearwall_index={clearwall_index} out of range for in_channels={in_channels}; "
                "DeltaFNO requires the enriched feature_set carrying the clear-wall theta channel."
            )
        self.clearwall_index = int(clearwall_index)
        if domain_padding is None:
            domain_padding = [0.25, 0.0]
        self.fno = FNO(
            n_modes=tuple(n_modes),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            domain_padding=list(domain_padding),
            positional_embedding=positional_embedding,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return ``theta_prior + fno(x)``; size-agnostic in H, W."""
        i = self.clearwall_index
        theta_prior = x[:, i : i + 1]
        return theta_prior + self.fno(x)


def build_delta_fno(
    in_channels: int = 6,
    out_channels: int = 1,
    n_modes: tuple[int, int] = (8, 16),
    hidden_channels: int = 32,
    n_layers: int = 4,
    clearwall_index: int = 3,
    domain_padding: list[float] | None = None,
    positional_embedding: str | nn.Module = "grid",
) -> nn.Module:
    """Construct a :class:`DeltaFNO`. See the class for argument semantics."""
    return DeltaFNO(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        clearwall_index=clearwall_index,
        domain_padding=domain_padding,
        positional_embedding=positional_embedding,
    )
