"""
losses.py — CycleGAN loss functions.

Three losses combine during training:

1. **Adversarial (GAN) loss** — makes generated images look realistic.
2. **Cycle-consistency loss** — enforces F(G(x)) ≈ x and G(F(y)) ≈ y,
   preventing mode collapse and preserving content structure.
3. **Identity loss** — optional regulariser that keeps colour / tone
   when the input is already in the target domain (G(y) ≈ y).

References
----------
Zhu et al., ICCV 2017 — https://arxiv.org/abs/1703.10593
Mao et al., ICCV 2017 — Least-squares GAN (LSGAN)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adversarial loss
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """
    Flexible GAN loss supporting vanilla (BCE) and least-squares (MSE) modes.

    Automatically creates real / fake target tensors on the correct device,
    so callers only pass the discriminator output map.

    Args:
        mode:  ``"lsgan"`` (default — L2, more stable) or ``"vanilla"`` (BCE).
        real_label: Value assigned to *real* targets (default 1.0).
        fake_label: Value assigned to *fake* targets (default 0.0).

    Example::

        criterion = GANLoss("lsgan").to(device)
        loss_G = criterion(D(fake_B), is_real=True)   # G wants D to say "real"
        loss_D_real = criterion(D(real_B), is_real=True)
        loss_D_fake = criterion(D(fake_B.detach()), is_real=False)
    """

    def __init__(
        self,
        mode: str = "lsgan",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("real_target", torch.tensor(real_label))
        self.register_buffer("fake_target", torch.tensor(fake_label))

        if mode == "lsgan":
            self.loss_fn = nn.MSELoss()
        elif mode == "vanilla":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported GAN mode: {mode!r}. Choose 'lsgan' or 'vanilla'.")

    # ------------------------------------------------------------------
    def _expand_target(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Return a target tensor broadcast to *prediction*'s shape."""
        value = self.real_target if is_real else self.fake_target  # type: ignore[attr-defined]
        return value.expand_as(prediction)

    # ------------------------------------------------------------------
    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Args:
            prediction: Raw discriminator output (patch map or scalar).
            is_real:    ``True`` → target = real_label; ``False`` → fake_label.

        Returns:
            Scalar loss tensor.
        """
        target = self._expand_target(prediction, is_real)
        return self.loss_fn(prediction, target)


# ---------------------------------------------------------------------------
# Cycle-consistency loss
# ---------------------------------------------------------------------------

def cycle_loss(
    real: torch.Tensor,
    reconstructed: torch.Tensor,
    weight: float = 10.0,
) -> torch.Tensor:
    """
    L1 cycle-consistency loss.

    Penalises ``|F(G(x)) − x|₁`` (and symmetrically for the other direction).

    Args:
        real:           Original image x  (B, C, H, W).
        reconstructed:  Cycle-recovered image F(G(x))  (B, C, H, W).
        weight:         Scalar multiplier λ (paper uses 10).

    Returns:
        Weighted scalar loss.
    """
    return weight * F.l1_loss(reconstructed, real)


# ---------------------------------------------------------------------------
# Identity loss
# ---------------------------------------------------------------------------

def identity_loss(
    real: torch.Tensor,
    identity_mapped: torch.Tensor,
    weight: float = 5.0,
) -> torch.Tensor:
    """
    Identity regularisation loss.

    When the generator G_{A→B} receives a real B image it should return
    that image unchanged: ``G_{A→B}(y) ≈ y``.

    This prevents the generator from making unnecessary colour / tone shifts
    and is especially helpful for photo-stylisation tasks.

    Args:
        real:             Real image from the target domain  (B, C, H, W).
        identity_mapped:  Generator output on real target image  (B, C, H, W).
        weight:           Scalar multiplier (paper uses λ/2 = 5).

    Returns:
        Weighted scalar loss.
    """
    return weight * F.l1_loss(identity_mapped, real)