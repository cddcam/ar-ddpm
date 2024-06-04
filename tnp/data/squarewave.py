from typing import Tuple

import torch

from .synthetic import (
    SyntheticGenerator,
    SyntheticGeneratorBimodalInput,
    SyntheticGeneratorUniformInput,
)


class SquareWaveGeneratorBase(SyntheticGenerator):
    def __init__(self, *, min_freq: float, max_freq: float, noise_std: float, **kwargs):
        super().__init__(**kwargs)

        self.min_freq = min_freq
        self.max_freq = max_freq
        self.noise_std = noise_std

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        # Sample a frequency.
        freq = self.sample_freq()

        # Sample a uniformly distributed (conditional on frequency) offset.
        sample = torch.rand((self.batch_size,))
        offset = sample / freq

        # Construct the sawtooth and add noise.
        f = torch.where(
            torch.floor(x @ freq[:, None, None] - offset[:, None, None]) % 2 == 0, 
            1.0, 0.0)
        y = f + self.noise_std * torch.randn_like(f)

        return y, None

    def sample_freq(self) -> torch.Tensor:
        # Sample frequency.
        freq = (
            torch.rand((self.batch_size,)) * (self.max_freq - self.min_freq)
            + self.min_freq
        )
        return freq


class SquareWaveGenerator(SquareWaveGeneratorBase, SyntheticGeneratorUniformInput):
    pass


class SquareWaveGeneratorBimodalInput(
    SquareWaveGeneratorBase, SyntheticGeneratorBimodalInput
):
    pass
