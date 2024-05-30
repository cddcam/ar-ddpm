from abc import ABC, abstractmethod
from scipy.optimize import fsolve
from typing import Optional, Union, List, Tuple
import numpy as np
import torch
import math
from .base import BaseScheduler
from diffusers.utils.torch_utils import randn_tensor
import torch.distributions as td



def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.

    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DDPMScheduler(BaseScheduler):
    """
    `DDPMScheduler` is a scheduler for AR-DDPMs.

    Args:
        num_train_timesteps (`int`, defaults to 5):
            The number of steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            An array of betas to pass directly to the constructor without using `beta_start` and `beta_end`.
        variance_type (`str`, defaults to `"fixed_small"`):
            Clip the variance when adding noise to the denoised sample. Choose from `fixed_small`, `fixed_small_log`,
            `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        variance_type: str = "learned",
        prediction_type: str = "mean",
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        elif beta_schedule == "constant":
            self.betas = torch.tensor([beta_start]*max(num_train_timesteps, 1))
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.prediction_type = prediction_type
        self.variance_type = variance_type

        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_train_timesteps
        self.custom_timesteps = False

    def _get_variance(self, t, predicted_variance=None, variance_type=None):

        if variance_type is None:
            variance_type = self.variance_type

        if variance_type == "learned":
            return predicted_variance
        
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            variance = variance
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_small_log":
            variance = torch.log(variance)
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = torch.log(current_beta_t)
        elif variance_type == "learned_range":
            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t)
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    # def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
    #     """
    #     "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
    #     prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
    #     s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
    #     pixels from saturation at each step. We find that dynamic thresholding results in significantly better
    #     photorealism as well as better image-text alignment, especially when using very large guidance weights."

    #     https://arxiv.org/abs/2205.11487
    #     """
    #     dtype = sample.dtype
    #     batch_size, channels, *remaining_dims = sample.shape

    #     if dtype not in (torch.float32, torch.float64):
    #         sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

    #     # Flatten sample for doing quantile calculation along each image
    #     sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

    #     abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

    #     s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
    #     s = torch.clamp(
    #         s, min=1, max=self.config.sample_max_value
    #     )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
    #     s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
    #     sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

    #     sample = sample.reshape(batch_size, channels, *remaining_dims)
    #     sample = sample.to(dtype)

    #     return sample

    def step(
        self,
        model_output: td.distribution.Distribution,
        timestep: int,
        sample: torch.FloatTensor = None,
        generator=None,
    ) -> Tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The discrete timestep in the diffusion chain for which we make the prediction.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.

        Returns:
            `tuple`:
                A tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        # If using non-diagonal covariance (i.e. InnerprodGaussianLikelihood)
        if not isinstance(model_output, td.Normal):
            # For the noised up variables - sample, for the unnoised variable - return mean
            # TODO: Shouldn't we add noise here?
            if t > 0:
                return model_output.rsample((1,))[0][..., None] # + variance
            return model_output.mean[..., None]

        if self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = model_output.mean, model_output.variance
        else:
            model_output = model_output.mean
            predicted_variance = None

        if self.prediction_type == "mean":
            pred_prev_sample = model_output
        else:
            # TODO: add different ways for model parameterisation; does not work at the moment.
            # 1. compute alphas, betas
            alpha_prod_t = self.alphas_cumprod[t]
            alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if self.prediction_type == "epsilon":
                pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif self.prediction_type == "sample":
                pred_original_sample = model_output
            elif self.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample` or"
                    " `v_prediction`  for the DDPMScheduler."
                )

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
            current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        # 6. Add noise
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Add noise according to x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise (Eq. (4) from DDPM)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def sample_timesteps(self, original_samples: torch.FloatTensor):
        device = original_samples.device
        timesteps = torch.randint(high=self.num_train_timesteps + 1, size=(1, 1), device=device)
        return timesteps.expand(original_samples.shape[0], original_samples.shape[1])
    
    def get_mu_var(self, 
                   original_samples: torch.FloatTensor, 
                   timesteps: torch.Tensor, 
                   noised_samples: torch.FloatTensor = None,
                   mask_targets: torch.LongTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Get the true posterior mean and variance when conditioning on the original
        samples and noised variables from a noise layer above 
        (i.e. moments of q(x_t|x_{t+1]}, x_0)) - Eq. (7) from DDPM

        Args:
            original_samples (`torch.FloatTensor`):
                The original samples (x_0).
            timesteps (`torch.Tensor`):
                The current discrete timestep in the diffusion chain.
            noised_samples (`torch.FloatTensor`, *optional*):
                The noised up samples from one noise layer above (x_{t+1}).
            mask_targets (`torch.LongTensor`, *optional*):
                Mask for the targets from the noise layer above.

        Returns:
            `tuple`:
                A tuple is returned where the first element is the mean and the second the variance.

        """
        
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # Append 1 for t=0 - this will affect the formulas for the multipliers below
        alphas_cumprod = torch.cat([torch.tensor([1.0], device=original_samples.device), 
                                    self.alphas_cumprod.to(dtype=original_samples.dtype)])
        timesteps = timesteps.to(original_samples.device)
        self.betas = self.betas.to(device=original_samples.device)
        betas = self.betas.to(dtype=original_samples.dtype)

        sqrt_alpha_prod = alphas_cumprod ** 0.5
        one_minus_alpha_prod = 1 - alphas_cumprod

        sqrt_alphas = (1 - betas) ** 0.5

        # If not predicting the variables at the highest level of diffusion 
        # (i.e. t=T where there are no noised up target values)
        if timesteps[0, 0] != self.num_train_timesteps:
            # Eq. (7) from DDPM, except that sqrt_alpha_prod and one_minus_alpha_prod have 
            # been shifted by 1 by appending 1 for t=0
            original_samples_multiplier = ((sqrt_alpha_prod[timesteps] * betas[timesteps]) /
                                        one_minus_alpha_prod[timesteps + 1])
            while len(original_samples_multiplier.shape) < len(original_samples.shape):
                original_samples_multiplier = original_samples_multiplier.unsqueeze(-1)

            noised_samples_multiplier = (sqrt_alphas[timesteps] * one_minus_alpha_prod[timesteps] /
                                        one_minus_alpha_prod[timesteps + 1])
            while len(noised_samples_multiplier.shape) < len(original_samples.shape):
                noised_samples_multiplier = noised_samples_multiplier.unsqueeze(-1)
            
            mu = original_samples_multiplier * original_samples + noised_samples_multiplier * noised_samples

            var = one_minus_alpha_prod[timesteps]/one_minus_alpha_prod[timesteps + 1] * betas[timesteps]

            # For the masked variables, compute mean and variance without conditioning
            # on the noised variables from the layer above (q(x_t|x_0))
            mu_masked = sqrt_alpha_prod[timesteps][..., None] * original_samples
            var_masked = one_minus_alpha_prod[timesteps]

            while len(var.shape) < len(mu.shape):
                var = var.unsqueeze(-1)

            while len(var_masked.shape) < len(mu.shape):
                var_masked = var_masked.unsqueeze(-1)

            mask_targets = mask_targets.float()[None]

            while len(mask_targets.shape) < len(mu.shape):
                mask_targets = mask_targets.unsqueeze(-1)

            # Final means and variances - use mu_masked for masked targets
            # and mu for the unmasked targets
            mu = mask_targets * mu + (1 - mask_targets) * mu_masked
            var = mask_targets * var + (1 - mask_targets) * var_masked
        else:
            # If predicting variables at highest level of diffusion
            # use forward diffusion formula (Eq. (4) from DDPM)
            mu = sqrt_alpha_prod[timesteps][..., None] * original_samples
            var = one_minus_alpha_prod[timesteps]

            while len(var.shape) < len(mu.shape):
                var = var.unsqueeze(-1)

        return mu, var
    
    def previous_timestep(self, timestep):
        if self.custom_timesteps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            num_inference_steps = (
                self.num_inference_steps if self.num_inference_steps else self.num_train_timesteps
            )
            prev_t = timestep - max(1, self.num_train_timesteps) // max(1, num_inference_steps)

        return prev_t


    # def get_velocity(
    #     self, sample: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
    # ) -> torch.FloatTensor:
    #     # Make sure alphas_cumprod and timestep have same device and dtype as sample
    #     self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
    #     alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
    #     timesteps = timesteps.to(sample.device)

    #     sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    #     sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    #     while len(sqrt_alpha_prod.shape) < len(sample.shape):
    #         sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    #     sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    #     sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    #     while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
    #         sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    #     velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    #     return velocity

    def __len__(self):
        return self.num_train_timesteps