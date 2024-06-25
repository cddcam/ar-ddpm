from typing import Tuple

import torch

from .synthetic import (
    SyntheticGenerator,
    SyntheticGeneratorMixture,
    SyntheticGeneratorUniformInput,
)


class DetPolynomialGeneratorBase(SyntheticGenerator):
    def __init__(self, *, coefficients: Tuple, noise_std: float, **kwargs):
        super().__init__(**kwargs)

        self.coefficients = coefficients
        self.noise_std = noise_std

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:

        # Compute the polynomial
        result = self.get_poly(x)
        
        # Add noise
        y = torch.normal(result, self.noise_std)

        return y, None, ("deterministic_polynomials", self.coefficients, self.noise_std)
    
    def get_poly(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        # Compute the mean based on the poylnomial coefficients
        # Convert coefficients to a tensor
        coefficients_tensor = torch.tensor(self.coefficients, dtype=x.dtype, device=x.device).view(1, 1, 1, -1)
        # Create a tensor of powers of x
        powers = torch.arange(len(self.coefficients), dtype=x.dtype, device=x.device).view(1, 1, 1, -1)
        # Compute the powers of x, resulting in shape [batch_size, num_points, len(coefficients)]
        x_powers = x.unsqueeze(-1) ** powers
        # Compute the polynomial
        result = (x_powers * coefficients_tensor).sum(dim=-1)
        
        return result


class DetPolynomialGenerator(DetPolynomialGeneratorBase, SyntheticGeneratorUniformInput):
    pass

class DetPolynomialGeneratorMixture(SyntheticGeneratorMixture):
    pass

