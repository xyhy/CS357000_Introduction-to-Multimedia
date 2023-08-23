import math

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from entropy_models_self import GaussianConditional
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from compressai.models.utils import update_registered_buffers

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with any number of
    EntropyBottleneck or GaussianConditional modules.
    """

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                )

            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                )

        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    def update(self, scale_table=None, force=False):
        """Updates EntropyBottleneck and GaussianConditional CDFs.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (torch.Tensor): table of scales (i.e. stdev)
                for initializing the Gaussian distributions
                (default: 64 logarithmically spaced scales from 0.11 to 256)
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if at least one of the modules was updated.
        """
        if scale_table is None:
            scale_table = get_scale_table()
        updated = False
        for _, module in self.named_modules():
            if isinstance(module, EntropyBottleneck):
                updated |= module.update(force=force)
            if isinstance(module, GaussianConditional):
                updated |= module.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self) -> Tensor:
        r"""Returns the total auxiliary loss over all ``EntropyBottleneck``\s.

        In contrast to the primary "net" loss used by the "net"
        optimizer, the "aux" loss is only used by the "aux" optimizer to
        update *only* the ``EntropyBottleneck.quantiles`` parameters. In
        fact, the "aux" loss does not depend on image data at all.

        The purpose of the "aux" loss is to determine the range within
        which most of the mass of a given distribution is contained, as
        well as its median (i.e. 50% probability). That is, for a given
        distribution, the "aux" loss converges towards satisfying the
        following conditions for some chosen ``tail_mass`` probability:

        * ``cdf(quantiles[0]) = tail_mass / 2``
        * ``cdf(quantiles[1]) = 0.5``
        * ``cdf(quantiles[2]) = 1 - tail_mass / 2``

        This ensures that the concrete ``_quantized_cdf``\s operate
        primarily within a finitely supported region. Any symbols
        outside this range must be coded using some alternative method
        that does *not* involve the ``_quantized_cdf``\s. Luckily, one
        may choose a ``tail_mass`` probability that is sufficiently
        small so that this rarely occurs. It is important that we work
        with ``_quantized_cdf``\s that have a small finite support;
        otherwise, entropy coding runtime performance would suffer.
        Thus, ``tail_mass`` should not be too small, either!
        """
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)




class ScaleHyperprior(CompressionModel):
    
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=N, kernel_size=3, stride=1, padding=1),
            GDN(N),
            nn.Conv2d(in_channels=N, out_channels=N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(in_channels=N, out_channels=N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(in_channels=N, out_channels=M, kernel_size=5, stride=2, padding=2),
        )

        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(in_channels=M, out_channels=N, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(in_channels=N, out_channels=N, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(in_channels=N, out_channels=N, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(in_channels=N, out_channels=1, kernel_size=3, stride=1, padding=1),
        )

        self.h_a = nn.Sequential(
            nn.Conv2d(in_channels=M, out_channels=N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=N, out_channels=N, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=N, out_channels=N, kernel_size=5, stride=2, padding=2),
        )

        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(in_channels=N, out_channels=N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=N, out_channels=N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=N, out_channels=M, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}