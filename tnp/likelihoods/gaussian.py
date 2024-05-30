import torch
import torch.distributions as td
from torch import nn
from typing import Tuple, Union, Optional

from .base import Likelihood


class NormalLikelihood(Likelihood):
    def __init__(self, noise: float, train_noise: bool = True):
        super().__init__()

        self.log_noise = nn.Parameter(
            torch.as_tensor(noise).log(), requires_grad=train_noise
        )

    @property
    def noise(self):
        return self.log_noise.exp()

    @noise.setter
    def noise(self, value: float):
        self.log_noise = nn.Parameter(torch.as_tensor(value).log())

    def forward(self, x: torch.Tensor) -> td.Normal:
        return td.Normal(x, self.noise)


class HeteroscedasticNormalLikelihood(Likelihood):
    def __init__(self, min_noise: float = 0.0):
        super().__init__()

        self.min_noise = min_noise

    def forward(self, x: torch.Tensor, t: Optional[int] = None) -> td.Normal:
        assert x.shape[-1] % 2 == 0

        loc, log_var = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        scale = (
            nn.functional.softplus(log_var) ** 0.5  # pylint: disable=not-callable
            + self.min_noise
        )
        return td.Normal(loc, scale)

class InnerprodGaussianLikelihood(Likelihood):
    
    def __init__(
            self,  
            covariance_feature_dim: int = 512,
            noise_type: str = 'homo',
            jitter: float = 1e-6,
            noise_params_dim: int = 0,
            ):
        
        super().__init__()
        
        # Noise type can be homoscedastic or heteroscedastic
        assert noise_type in ["homo", "hetero", "noiseless"]

        self.jitter = jitter        
        # Set noise type, initialise noise variable if necessary
        self.noise_type = noise_type
        
        if self.noise_type == "homo":
            self.noise_unconstrained = nn.Parameter(torch.tensor([0.]*noise_params_dim))
        
        # Compute total number of features expected by layer
        self.mean_dim = 1
        self.num_embedding = covariance_feature_dim
        
        self.num_features = self.mean_dim        + \
                            self.num_embedding
        
        
    def _mean_and_cov(
            self, 
            x: torch.Tensor, 
            t: Optional[int] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] :
        """
        Computes mean and covariance of linear Gaussian layer, as specified
        by the parameters in *x*. This method may give covariances
        which are close to singular, so the method mean_and_cov should be
        used instead.
        
        Arguments:
            x : torch.tensor, (B, T, C)
            t: torch.tensor, (B, T)
            
        Returns:
            mean  : torch.tensor, (B, T)
            f_cov : torch.tensor, (B, T, T)
            y_cov : torch.tensor, (B, T, T)
        """
        
        # Check tensor has three dimensions, and last dimension has size self.num_features
        assert (len(x.shape) == 3) and \
               (x.shape[2] == self.num_features)
        
        # Batch and datapoint dimensions
        B, T, C = x.shape
        
        # Compute mean vector
        mean = x[:, :, 0]

        jitter = torch.diag_embed(torch.tensor(self.jitter).repeat(B, T).to(x.device))
        
        # Slice out components of covariance - z and noise
        if self.noise_type == "homo":
            # Basis functions are the remaining dimensions
            z = x[:, :, 1:] / C**0.5
            
            noise = torch.nn.Softplus()(self.noise_unconstrained[t])
            noise = noise[None, None].repeat(B, T)
            noise = torch.diag_embed(noise)
            
        elif self.noise_type == 'hetero':
            # Basis functions are the remaining (dimensions - 1)
            z = x[:, :, 1:-1] / C**0.5
            
            # Use last dimension for heteroscedastic noise
            noise = torch.nn.Softplus()(x[:, :, -1])
            noise = torch.diag_embed(noise)
        else:
            noise = 0.0
        
        # Covariance is the product of the basis functions
        f_cov = torch.einsum("bnc, bmc -> bnm", z, z)
        y_cov = f_cov + noise + jitter
        
        return mean, f_cov, y_cov
    
    
    def forward(
            self, 
            x: torch.Tensor, 
            t: Optional[int] = None
        ) -> Union[td.MultivariateNormal, td.LowRankMultivariateNormal]:
        # Check tensor has three dimensions, and last dimension has size num_features
        assert (len(x.shape) == 3) and \
               (x.shape[2] == self.num_features)
        
        B, T, C = x.shape
        
        # If num datapoints smaller than num embedding, return full-rank
        if T - 1 <= self.num_embedding:
            
            mean, f_cov, y_cov = self._mean_and_cov(x, t)
            cov = f_cov if self.noise_type == 'noiseless' else y_cov
            
            dist = td.MultivariateNormal(loc=mean, covariance_matrix=cov)
            return dist
          
        # Otherwise, return low-rank 
        else:  
            # Split tensor into mean and embedding
            mean = x[:, :, 0]
            z = x[:, :, 1:-1] / C**0.5
            
            jitter = torch.tensor(self.jitter).repeat(B, T).to(z.device)

            if self.noise_type == 'noiseless':
                noise = 0.0
                
            elif self.noise_type == "homo":
                noise = torch.nn.Softplus()(self.noise_unconstrained[t])
                noise = noise[None, None].repeat(B, T)

            else:
                noise = torch.nn.Softplus()(x[:, :, -1])
                
            noise = noise + jitter
            
            dist = td.LowRankMultivariateNormal(
                loc=mean,
                cov_factor=z,
                cov_diag=noise
            )
            return dist
