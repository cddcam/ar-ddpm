import argparse
import os
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import einops
import hiyapyco
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm
import numpy as np
import torch.distributions as td

import wandb

from ..data.base import Batch, DataGenerator
from ..data.synthetic import SyntheticBatch
from ..models.base import ConditionalNeuralProcess, NeuralProcess
from ..models.convcnp import GriddedConvCNP
from ..utils.batch import compress_batch_dimensions
from .initialisation import weights_init
from ..schedulers.base import BaseScheduler


class ModelCheckpointer:
    def __init__(self, checkpoint_dir: Optional[str] = None, logging: bool = True):
        self.logging = logging

        self.checkpoint_dir: Optional[str] = None

        if checkpoint_dir is None and logging:
            checkpoint_dir = f"{wandb.run.dir}/checkpoints"

            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)

            self.checkpoint_dir = checkpoint_dir

        self.best_validation_loss = float("inf")

    def update_best_and_last_checkpoint(
        self,
        model: nn.Module,
        val_result: Dict[str, torch.Tensor],
        prefix: Optional[str] = None,
        update_last: bool = True,
    ) -> None:
        """Update the best and last checkpoints of the model.

        Arguments:
            model: model to save.
            val_result: validation result dictionary.
        """

        loss_ci = val_result["mean_loss"]

        if loss_ci < self.best_validation_loss:
            self.best_validation_loss = loss_ci
            if self.logging:
                assert self.checkpoint_dir is not None
                if prefix is not None:
                    name = f"{prefix}best.ckpt"
                else:
                    name = "best.ckpt"

                torch.save(
                    model.state_dict(),
                    os.path.join(self.checkpoint_dir, name),
                )

        if update_last and self.logging:
            assert self.checkpoint_dir is not None
            torch.save(
                model.state_dict(), os.path.join(self.checkpoint_dir, "last.ckpt")
            )

    def load_checkpoint(
        self, model: nn.Module, checkpoint: str = "last", path: Optional[str] = None
    ) -> None:
        if path is None and self.logging:
            assert self.checkpoint_dir is not None
            path = os.path.join(self.checkpoint_dir, f"{checkpoint}.ckpt")
        elif path is None:
            raise ValueError("Not logging to any checkpoints.")

        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"Checkpoint file {path} not found.")

def discrete_denoising_loss_fn(
    model: nn.Module, 
    batch: Batch, 
    scheduler: BaseScheduler,
    subsample_targets: bool = False,
) -> torch.Tensor:
    """Perform a single DDPM training step, returning the loss.

    Arguments:
        model: model to train.
        batch: batch of data.
        scheduler: DDPM scheduler.
        subsample_targets: whether to subsample the targets at each noise level.

    Returns:
        loss: average negative KL Div.
    """
    # Randomly sample diffusion time for targets and set to 0 for context
    tt = scheduler.sample_timesteps(batch.xt)
    tc = torch.zeros(batch.xc.shape[0], batch.xc.shape[1], device=batch.xc.device)

    if tt[0, 0] != len(scheduler):
        # For all noise levels except highest one, sample noised up targets
        # from the noise level above  (tt + 1)
        noise = torch.randn(batch.yt.shape, device=batch.yt.device)
        noised_targets = scheduler.add_noise(batch.yt, noise, tt)
        # TODO: Shouldn't this be ?
        # noised_targets = scheduler.add_noise(batch.yt, noise, tt + 1)
        if subsample_targets:
            # Only predict based on a subset of noised up targets
            x_prob = np.random.uniform()
            mask_targets = torch.ones(batch.xt.shape[1], device=batch.xt.device)
            mask_targets = torch.bernoulli(mask_targets * x_prob) == 1
        else:
            # Predict based on all noised up targets from level above
            mask_targets = torch.ones(batch.xt.shape[1], device=batch.xt.device) == 1
        
        # Context becomes unnoised context + previous noised up targets 
        # (from noise level above, hence we append tt[:, mask_targets]+1)
        yc = torch.cat([batch.yc, noised_targets[:, mask_targets]], dim=1)
        xc = torch.cat([batch.xc, batch.xt[:, mask_targets]], dim=1)
        tc = torch.cat([tc, tt[:, mask_targets]+1], dim=1)
    else:
        # At the highest level of diffusion there are no noised up targets
        # to condition on (no noise level above), so just condition on context
        xc = batch.xc
        yc = batch.yc
        noised_targets = None
        mask_targets = None

    # Get predictive distribution for targets at noise level tt
    pred_dist = model(xc, yc, batch.xt, tc, tt)
    # Get true posterior when conditioning on true data
    int_loc, int_var = scheduler.get_mu_var(batch.yt, tt, noised_targets, mask_targets)

    if isinstance(pred_dist, td.Normal):
        # If covariance is diagonal
        kl_div = pred_dist.log_prob(int_loc).sum() / int_loc.numel()
        kl_div -= 0.5 * (int_var/(pred_dist.variance + 1e-7)).sum() / int_var.numel()
    else:
        # If covariance is non-diagonal (e.g. as in InnerprodGaussianLikelihood)
        kl_div = pred_dist.log_prob(int_loc[..., 0]).sum() / int_loc.numel()
        kl_div -= 0.5 * (int_var[..., 0]* torch.diagonal(pred_dist.precision_matrix, dim1=-2, dim2=-1)).sum() / int_loc.numel()
    return -kl_div


def discrete_denoising_pred_fn(
    model: nn.Module, 
    batch: Batch, 
    scheduler: BaseScheduler,
    x_plot: Optional[torch.FloatTensor] = None,
    subsample_targets: bool = False,
)  -> Union[Tuple[torch.distributions.Distribution, torch.FloatTensor, torch.FloatTensor], 
            Tuple[torch.distributions.Distribution, torch.distributions.Distribution, torch.FloatTensor, torch.FloatTensor]]: 
    """Perform a single DDPM prediction step, returning the model distribution.

    Arguments:
        model: model to train.
        batch: batch of data.
        scheduler: DDPM scheduler.
        x_plot: x locations for plotting.
        subsample_targets: whether to subsample the targets at each noise level.

    Returns:
        pred_dist_target: Predictive distribution at target locations.
        pred_dist_plot (optional): Predictive distribution at plotting locations.
        noised_targets: Sampled noised up targets at the current diffusion timestep.
        tt[:, 0]: The diffusion timestep.
    """
    # Randomly sample diffusion time for targets and set to 0 for context
    tt = scheduler.sample_timesteps(batch.xt)
    tc = torch.zeros(batch.xc.shape[0], batch.xc.shape[1], device=batch.xc.device)

    if tt[0, 0] != len(scheduler):
        # For all noise levels except highest one, sample noised up targets
        # from the noise level above  (tt + 1)
        noise = torch.randn(batch.yt.shape, device=batch.yt.device)
        noised_targets = scheduler.add_noise(batch.yt, noise, tt)
        if subsample_targets:
            # Only predict based on a subset of noised up targets
            x_prob = np.random.uniform()
            mask_targets = torch.ones(batch.xt.shape[1], device=batch.xt.device)
            mask_targets = torch.bernoulli(mask_targets * x_prob) == 1
        else:
            # Predict based on all noised up targets from level above
            mask_targets = torch.ones(batch.xt.shape[1], device=batch.xt.device) == 1
        
        # Context becomes unnoised context + previous noised up targets 
        # (from noise level above, hence we append tt[:, mask_targets]+1)
        yc = torch.cat([batch.yc, noised_targets[:, mask_targets]], dim=1)
        xc = torch.cat([batch.xc, batch.xt[:, mask_targets]], dim=1)
        tc = torch.cat([tc, tt[:, mask_targets]+1], dim=1)

    else:
        # At the highest level of diffusion there are no noised up targets
        # to condition on (no noise level above), so just condition on context
        xc = batch.xc
        yc = batch.yc
        noised_targets = None
        mask_targets = None

    # TODO: Why do we need this here? We are not using mask_targets from here on
    if mask_targets is not None and mask_targets.sum() == 0:
        mask_targets = None
        
    pred_dist_target = model(xc, yc, batch.xt, tc, tt)
    
    # Plotting
    if x_plot is not None:
        tt_plot = tt[:, :1].expand(-1, x_plot.shape[1])
        pred_dist_plot = model(xc, yc, x_plot, tc, tt_plot)
        return pred_dist_target, pred_dist_plot, noised_targets, tt[:, 0]
    return pred_dist_target, noised_targets, tt[:, 0] 


def discrete_denoising_sampling(
        model: nn.Module, 
        batch: Batch, 
        scheduler: BaseScheduler,
        x_plot: Optional[torch.FloatTensor] = None,
):
    """Sample from the diffusion process, starting from the highest noise level.

    Arguments:
        model: model to train.
        batch: batch of data.
        scheduler: DDPM scheduler.
        x_plot: x locations for plotting.
        subsample_targets: whether to subsample the targets at each noise level.

    Returns:
        noised_samples_history: The noised up samples from t=T up to t=0.
        plot_distributions: Distributions for plotting.
    """
    # Context points are always unnoised (t=0)
    # Start diffusion process for the target points from the highest noise level (t=T)
    tc = torch.zeros(batch.xc.shape[0], batch.xc.shape[1], device=batch.xc.device, dtype=torch.int)
    tt = torch.ones(batch.xt.shape[0], batch.xt.shape[1], device=batch.xt.device, dtype=torch.int) * len(scheduler)

    # Get predictive distribution of targets at the highest noise level t=T
    pred_dist = model(batch.xc, batch.yc, batch.xt, tc, tt)

    noised_samples = scheduler.step(pred_dist, tt[0, 0])

    noised_samples_history = [noised_samples.clone()]
    plot_distributions = []

    if x_plot is not None:
        tt_plot = torch.ones(batch.xt.shape[0], x_plot.shape[1], device=batch.xt.device) * len(scheduler)
        pred_dist_plot = model(batch.xc, batch.yc, x_plot, tc, tt_plot)
        plot_distributions += [pred_dist_plot]

    # Predict at each diffusion timestep, starting from t=T-1 up to t=0 (inclusive)
    for t in range(len(scheduler) - 1, -1, -1):
        tt = torch.ones(batch.xt.shape[0], batch.xt.shape[1], device=batch.xt.device, dtype=torch.int) * t
        # Context becomes unnoised context + previous noised up targets 
        # (from noise level above, hence we append tt + 1)
        yc_t = torch.cat([batch.yc, noised_samples], dim=1)
        xc_t = torch.cat([batch.xc, batch.xt], dim=1)
        tc_t = torch.cat([tc, tt+1], dim=1)
        
        # Get prediction of noised targets at the current diffusion timestep
        pred_dist = model(xc_t, yc_t, batch.xt, tc_t, tt)
        noised_samples = scheduler.step(pred_dist, tt[0, 0])
        noised_samples_history += [noised_samples.clone()]

        if x_plot is not None:
            tt_plot = torch.ones(batch.xt.shape[0], x_plot.shape[1], device=batch.xt.device, dtype=torch.int) * t
            pred_dist_plot = model(xc_t, yc_t, x_plot, tc_t, tt_plot)
            plot_distributions += [pred_dist_plot]
    return noised_samples_history, plot_distributions


def discrete_denoising_loglik(
        model: nn.Module, 
        batch: Batch, 
        scheduler: BaseScheduler,
        num_samples: int = 100,
        split_batch: bool = False,
):
    """Compute log-likelihood .

    Arguments:
        model: model to train.
        batch: batch of data.
        scheduler: DDPM scheduler.
        num_samples: Number of samples used in MCMC sampling.
        split_batch: whether to split the batch.

    Returns:
        loglik: Average log-likelihood.
        loglik_joint: Average joint log-likelihood.
    """
    # No diffusion
    if len(scheduler) == 0:
        tt = torch.zeros(batch.xt.shape[0], batch.xt.shape[1], device=batch.xc.device)
        tc = torch.zeros(batch.xc.shape[0], batch.xc.shape[1], device=batch.xc.device)

        pred_dist = model(batch.xc, batch.yc, batch.xt, tc, tt)

        if isinstance(pred_dist, td.Normal):
            # If covariance is diagonal
            loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
            return loglik, None
        else:
            # If covariance is non-diagonal (e.g. as in InnerprodGaussianLikelihood),
            # diagnolise it to get the marginal for each target point
            std = torch.diagonal(pred_dist.covariance_matrix, dim1=-2, dim2=-1).sqrt()
            loglik = td.Normal(loc=pred_dist.mean, scale=std).log_prob(batch.yt[..., 0]).sum() / batch.yt[..., 0].numel()

            # For the joint distribution, use the non-diagonal covariance
            loglik_joint = pred_dist.log_prob(batch.yt[..., 0]).sum() / batch.yt[0, ..., 0].numel()
            return loglik, loglik_joint

    if split_batch:
        logliks = []
        logliks_joint = None
        # Split the batch
        for i in range(0, batch.xc.shape[0]):
            tc = torch.zeros(num_samples, batch.xc.shape[1], device=batch.xc.device, dtype=torch.int)
            tt = torch.ones(
                num_samples, 
                batch.xt.shape[1], 
                device=batch.xt.device, 
                dtype=torch.int) * len(scheduler)
            
            # Extract context and target for the current batch
            xc = batch.xc[i: i + 1].expand(num_samples, *((-1,)*(len(batch.xc.shape) - 1)))
            yc = batch.yc[i: i + 1].expand(num_samples, *((-1,)*(len(batch.yc.shape) - 1)))
            xt = batch.xt[i: i + 1].expand(num_samples, *((-1,)*(len(batch.xt.shape) - 1)))
            yt = batch.yt[i: i + 1].expand(num_samples, *((-1,)*(len(batch.yt.shape) - 1)))

            # Get predictive distribution for the noised targets at the highest noise level (t=T)
            pred_dist = model(xc, yc, xt, tc, tt)

            noised_samples = scheduler.step(pred_dist, tt[0, 0])

            # Predict at each diffusion timestep, starting from t=T-1 up to t=0 (inclusive)
            for t in range(len(scheduler) - 1, -1, -1):
                tt = torch.ones(
                    num_samples, 
                    batch.xt.shape[1], 
                    device=batch.xt.device, 
                    dtype=torch.int) * t
                # Context becomes unnoised context + previous noised up targets 
                # (from noise level above, hence we append tt + 1)
                yc_t = torch.cat([yc, noised_samples], dim=1)
                xc_t = torch.cat([xc, xt], dim=1)
                tc_t = torch.cat([tc, tt+1], dim=1)
                
                pred_dist = model(xc_t, yc_t, xt, tc_t, tt)
                noised_samples = scheduler.step(pred_dist, tt[0, 0])

            if isinstance(pred_dist, td.Normal):
                loglik = pred_dist.log_prob(yt).sum(dim=list(range(1, len(yt.shape))))
                loglik = (torch.logsumexp(loglik, dim=0) - np.log(float(num_samples))) / yt[0, ..., 0].numel()
                loglik_joint = None
            else:
                std = torch.diagonal(pred_dist.covariance_matrix, dim1=-2, dim2=-1).sqrt()
                loglik = td.Normal(loc=pred_dist.mean, scale=std).log_prob(yt[..., 0])
                loglik = loglik.sum(dim=list(range(1, len(yt[..., 0].shape))))
                loglik = (torch.logsumexp(loglik, dim=0) - np.log(float(num_samples))) / yt[0, ..., 0].numel()

                loglik_joint = pred_dist.log_prob(yt[..., 0])
                loglik_joint = (torch.logsumexp(loglik_joint, dim=0) - np.log(float(num_samples))) / yt[0, ..., 0].numel()
            logliks.append(loglik)
            if loglik_joint is not None:
                if logliks_joint is None:
                    logliks_joint = [loglik_joint]
                else:
                    logliks_joint.append(loglik_joint)
        if logliks_joint is not None:
            logliks_joint = torch.stack(logliks_joint).mean()
        logliks = torch.stack(logliks).mean()
        return logliks, logliks_joint

    else:
        tc = torch.zeros(batch.xc.shape[0]*num_samples, batch.xc.shape[1], device=batch.xc.device, dtype=torch.int)
        tt = torch.ones(
            batch.xt.shape[0]*num_samples, 
            batch.xt.shape[1], 
            device=batch.xt.device, 
            dtype=torch.int) * len(scheduler)
        
        xc = batch.xc[:, None].expand(-1, num_samples, *((-1,)*(len(batch.xc.shape) - 1))).reshape(-1, *batch.xc.shape[1:])
        yc = batch.yc[:, None].expand(-1, num_samples, *((-1,)*(len(batch.yc.shape) - 1))).reshape(-1, *batch.yc.shape[1:])
        xt = batch.xt[:, None].expand(-1, num_samples, *((-1,)*(len(batch.xt.shape) - 1))).reshape(-1, *batch.xt.shape[1:])
        yt = batch.yt[:, None].expand(-1, num_samples, *((-1,)*(len(batch.yt.shape) - 1))).reshape(-1, *batch.yt.shape[1:])

        # Get predictive distribution for the noised targets at the highest noise level (t=T)
        pred_dist = model(xc, yc, xt, tc, tt)

        noised_samples = scheduler.step(pred_dist, tt[0, 0])

        # Predict at each diffusion timestep, starting from t=T-1 up to t=0 (inclusive)
        for t in range(len(scheduler) - 1, -1, -1):
            tt = torch.ones(
                batch.xt.shape[0]*num_samples, 
                batch.xt.shape[1], 
                device=batch.xt.device, 
                dtype=torch.int) * t
            # Context becomes unnoised context + previous noised up targets 
            # (from noise level above, hence we append tt + 1)
            yc_t = torch.cat([yc, noised_samples], dim=1)
            xc_t = torch.cat([xc, xt], dim=1)
            tc_t = torch.cat([tc, tt+1], dim=1)
            
            pred_dist = model(xc_t, yc_t, xt, tc_t, tt)
            noised_samples = scheduler.step(pred_dist, tt[0, 0])

        if isinstance(pred_dist, td.Normal):
            loglik = pred_dist.log_prob(yt).sum(dim=list(range(1, len(yt.shape))))
            loglik_joint = None
        else:
            loglik_joint = pred_dist.log_prob(yt[..., 0])  
            loglik_joint = loglik_joint.reshape(-1, num_samples)
            # Is this needed twice?
            loglik_joint = loglik_joint.reshape(-1, num_samples)
            loglik_joint = (torch.logsumexp(loglik_joint, dim=1) - np.log(float(num_samples))) / yt[0, ..., 0].numel() 
            loglik_joint = loglik_joint.mean()

            std = torch.diagonal(pred_dist.covariance_matrix, dim1=-2, dim2=-1).sqrt()
            loglik = td.Normal(loc=pred_dist.mean, scale=std).log_prob(yt[..., 0])
            loglik = loglik.sum(dim=list(range(1, len(yt[..., 0].shape))))     

        loglik = loglik.reshape(-1, num_samples)
        loglik = (torch.logsumexp(loglik, dim=1) - np.log(float(num_samples))) / yt[0, ..., 0].numel()
        loglik = loglik.mean()

        return loglik, loglik_joint

def np_pred_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.distributions.Distribution:
    if isinstance(model, GriddedConvCNP):
        pred_dist = model(mc=batch.mc_grid, y=batch.y_grid, mt=batch.mt_grid)
    elif isinstance(model, ConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)
    elif isinstance(model, NeuralProcess):
        pred_dist = model(
            xc=batch.xc, yc=batch.yc, xt=batch.xt, num_samples=num_samples
        )
    else:
        raise ValueError

    return pred_dist


def np_loss_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.Tensor:
    """Perform a single training step, returning the loss, i.e.
    the negative log likelihood.

    Arguments:
        model: model to train.
        batch: batch of data.

    Returns:
        loss: average negative log likelihood.
    """
    pred_dist = np_pred_fn(model, batch, num_samples)
    loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()

    return -loglik


def train_epoch(
    model: nn.Module,
    scheduler: BaseScheduler,
    generator: DataGenerator,
    optimiser: torch.optim.Optimizer,
    step: int,
    loss_fn: Callable = np_loss_fn,
    gradient_clip_val: Optional[float] = None,
    subsample_targets: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    epoch = tqdm(generator, total=len(generator), desc="Training")
    losses = []
    for batch in epoch:
        optimiser.zero_grad()
        loss = loss_fn(model=model, batch=batch, scheduler=scheduler, subsample_targets=subsample_targets)
        loss.backward()

        if gradient_clip_val is not None:
            nn.utils.clip_grad_norm(model.parameters(), gradient_clip_val)

        optimiser.step()

        losses.append(loss.detach())
        epoch.set_postfix({"train/loss": loss.item()})

        if wandb.run is not None:
            wandb.log({"train/loss": loss, "step": step})

        step += 1

    loglik = -torch.stack(losses)
    train_result = {
        "loglik": loglik,
        "mean_loglik": loglik.mean(),
        "std_loglik": loglik.std() / (len(losses) ** 0.5),
        "mean_loss": -loglik.mean(),
        "std_loss": loglik.std() / (len(losses) ** 0.5),
    }

    return step, train_result


def val_epoch(
    model: nn.Module,
    scheduler: BaseScheduler,
    generator: DataGenerator,
    loss_fn: Callable = np_loss_fn,
    subsample_targets: bool = False,
) -> Tuple[Dict[str, Any], List[Batch]]:
    result = defaultdict(list)
    batches = []

    model.eval()

    for batch in tqdm(generator, total=len(generator), desc="Validation"):
        batches.append(batch)

        with torch.no_grad():
            loglik = -loss_fn(model=model, batch=batch, scheduler=scheduler, subsample_targets=subsample_targets)
        result["loglik"].append(loglik)

    loglik = torch.stack(result["loglik"])
    result["mean_loglik"] = loglik.mean()
    result["std_loglik"] = loglik.std() / (len(loglik) ** 0.5)
    result["mean_loss"] = -loglik.mean()
    result["std_loss"] = loglik.std() / (len(loglik) ** 0.5)

    return result, batches


def test_epoch(
    model: nn.Module,
    scheduler: BaseScheduler,
    generator: DataGenerator,
    loglik_fn: Callable = np_loss_fn,
    num_samples: int = 1,
    split_batch: bool = False,
) -> Tuple[Dict[str, Any], List[Batch]]:
    result = defaultdict(list)
    batches = []

    model.eval()

    for batch in tqdm(generator, total=len(generator), desc="Test"):
        batches.append(batch)

        with torch.no_grad():
            loglik, loglik_joint = loglik_fn(
                model=model, 
                batch=batch, 
                scheduler=scheduler, 
                num_samples=num_samples, 
                split_batch=split_batch,
                )

        if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
            gt_mean, gt_std, gt_loglik, gt_loglik_joint = batch.gt_pred(
                xc=batch.xc,
                yc=batch.yc,
                xt=batch.xt,
                yt=batch.yt,
            )

            gt_loglik = gt_loglik.sum() / batch.yt[..., 0].numel()
            gt_loglik_joint = gt_loglik_joint.sum() / batch.yt[..., 0].numel()

            result["gt_mean"].append(gt_mean)
            result["gt_std"].append(gt_std)
            result["gt_loglik"].append(gt_loglik)
            result["gt_loglik_joint"].append(gt_loglik_joint)

        result["loglik"].append(loglik)
        if loglik_joint is not None:
            result["loglik_joint"].append(loglik_joint)

    loglik = torch.stack(result["loglik"])
    result["mean_loglik"] = loglik.mean()
    result["std_loglik"] = loglik.std() / (len(loglik) ** 0.5)
    result["mean_loss"] = -loglik.mean()
    result["std_loss"] = loglik.std() / (len(loglik) ** 0.5)

    if "loglik_joint" in result:
        loglik_joint = torch.stack(result["loglik_joint"])
        result["mean_loglik_joint"] = loglik_joint.mean()
        result["std_loglik_joint"] = loglik_joint.std() / (len(loglik_joint) ** 0.5)

    if "gt_loglik" in result:
        gt_loglik = torch.stack(result["gt_loglik"])
        result["mean_gt_loglik"] = gt_loglik.mean()
        result["std_gt_loglik"] = gt_loglik.std() / (len(gt_loglik) ** 0.5)
        gt_loglik_joint = torch.stack(result["gt_loglik_joint"])
        result["mean_gt_loglik_joint"] = gt_loglik_joint.mean()
        result["std_gt_loglik_joint"] = gt_loglik_joint.std() / (len(gt_loglik_joint) ** 0.5)

    return result, batches


def create_default_config() -> DictConfig:
    default_config = {
        "misc": {
            "resume_from_checkpoint": None,
            "resume_from_path": None,
            "override_config": False,
            "plot_ar_mode": False,
            "logging": True,
            "seed": 0,
            "plot_interval": 1,
            "lightning_eval": False,
            "num_plots": 5,
            "gradient_clip_val": None,
            "only_plots": False,
            "savefig": False,
            "subplots": True,
            "loss_fn": {
                "_target_": "tnp.utils.experiment_utils.discrete_denoising_loss_fn",
                "_partial_": True,
            },
            "pred_fn": {
                "_target_": "tnp.utils.experiment_utils.discrete_denoising_pred_fn",
                "_partial_": True,
            },
        }
    }
    return OmegaConf.create(default_config)


def extract_config(
    raw_config: Union[str, Dict],
    config_changes: Optional[List[str]] = None,
    combine_default: bool = True,
) -> Tuple[DictConfig, Dict]:
    """Extract the config from the config file and the config changes.

    Arguments:
        config_file: path to the config file.
        config_changes: list of config changes.

    Returns:
        config: config object.
        config_dict: config dictionary.
    """
    # Register eval.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    if isinstance(raw_config, str):
        config = OmegaConf.load(raw_config)
    else:
        config = OmegaConf.create(raw_config)

    if combine_default:
        default_config = create_default_config()
        config = OmegaConf.merge(default_config, config)

    config_changes = OmegaConf.from_cli(config_changes)
    config = OmegaConf.merge(config, config_changes)
    config_dict = OmegaConf.to_container(config, resolve=True)

    return config, config_dict


def deep_convert_dict(layer: Any):
    to_ret = layer
    if isinstance(layer, OrderedDict):
        to_ret = dict(layer)

    try:
        for key, value in to_ret.items():
            to_ret[key] = deep_convert_dict(value)
    except AttributeError:
        pass

    return to_ret


def initialize_experiment() -> Tuple[DictConfig, ModelCheckpointer]:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--generator_config", type=str)
    parser.add_argument("--scheduler_config", type=str)
    args, config_changes = parser.parse_known_args()

    if args.generator_config is not None and args.scheduler_config is not None:
        # Merge generator config with config.
        raw_config = deep_convert_dict(
            hiyapyco.load(
                (args.config, args.generator_config, args.scheduler_config),
                method=hiyapyco.METHOD_MERGE,
                usedefaultyamlloader=True,
            )
        )
    elif args.generator_config is not None:
        raw_config = deep_convert_dict(
            hiyapyco.load(
                (args.config, args.generator_config),
                method=hiyapyco.METHOD_MERGE,
                usedefaultyamlloader=True,
            )
        )
    elif args.scheduler_config is not None:
        raw_config = deep_convert_dict(
            hiyapyco.load(
                (args.config, args.scheduler_config),
                method=hiyapyco.METHOD_MERGE,
                usedefaultyamlloader=True,
            )
        )
    else:
        raw_config = args.config

    # Initialise experiment, make path.
    config, config_dict = extract_config(raw_config, config_changes)

    # Get run and potentially override config before instantiation.
    if config.misc.resume_from_checkpoint is not None:
        # Downloads to "./checkpoints/last.ckpt".
        api = wandb.Api()
        run = api.run(config.misc.resume_from_checkpoint)

        # Overide config if specified.
        if config.misc.override_config:
            config = OmegaConf.create(run.config)
            config_dict = run.config

    # Instantiate experiment and load checkpoint.
    pl.seed_everything(config.misc.seed)
    experiment = instantiate(config)
    pl.seed_everything(experiment.misc.seed)

    if isinstance(experiment.model, nn.Module):
        if experiment.misc.resume_from_checkpoint:
            # Downloads to "./checkpoints/last.ckpt".
            ckpt_file = run.files("checkpoints/last.ckpt")[0]
            ckpt_file.download(replace=True)
            experiment.model.load_state_dict(
                torch.load("checkpoints/last.ckpt", map_location="cpu"), strict=True
            )

        elif experiment.misc.resume_from_path is not None:
            experiment.model.load_state_dict(
                torch.load(experiment.misc.resume_from_path, map_location="cpu"),
                strict=True,
            )

        else:
            # Initialise model weights.
            weights_init(experiment.model)
    else:
        print("Did not initialise as not nn.Module.")

    # Initialise wandb. Set logging: True if wandb logging needed.
    if experiment.misc.logging:
        wandb.init(
            project=experiment.misc.project,
            name=experiment.misc.name,
            config=config_dict,
        )

    checkpointer = ModelCheckpointer(logging=experiment.misc.logging)

    return experiment, checkpointer


def initialize_evaluation() -> DictConfig:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument(
        "--ckpt", type=str, choices=["val_best", "train_best", "last"], default="last"
    )
    args, config_changes = parser.parse_known_args()

    api = wandb.Api()
    run = api.run(args.run_path)

    # Initialise evaluation, make path.
    config, _ = extract_config(args.config, config_changes)

    # Set model to run.config.model.
    config.model = run.config["model"]
    config.scheduler = run.config["scheduler"]

    # Set random seed.
    pl.seed_everything(config.misc.seed)

    # Instantiate.
    experiment = instantiate(config)

    # Set random seed.
    pl.seed_everything(config.misc.seed)

    # Downloads to "./checkpoints/last.ckpt"
    ckpt_file = run.files(f"checkpoints/{args.ckpt}.ckpt")[0]
    ckpt_file.download(replace=True)

    experiment.model.load_state_dict(
        torch.load(f"checkpoints/{args.ckpt}.ckpt", map_location="cpu"), strict=True
    )

    # Initialise wandb.
    wandb.init(
        resume="must",
        project=run.project,
        name=run.name,
        id=run.id,
    )

    return experiment


def evaluation_summary(name: str, result: Dict[str, Any]) -> None:
    if wandb.run is None:
        return

    if "mean_loglik" in result:
        wandb.log({f"{name}/loglik": result["mean_loglik"]})

    if "mean_gt_loglik" in result:
        wandb.log(
            {
                f"{name}/gt_loglik": result["mean_gt_loglik"],
            }
        )

# TODO: Check with Matt which one we should use
def ar_predict(
    model: nn.Module,
    xc: torch.Tensor,
    yc: torch.Tensor,
    xt: torch.Tensor,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples from the joint predictive probability distribution. Assumes order of xt is given.

    Args:
        model (nn.Module): NeuralProcess model.
        xc (torch.Tensor): Context inputs.
        yc (torch.Tensor): Context outputs.
        xt (torch.Tensor): Target inputs.
        yt (torch.Tensor): Target outputs.
        num_samples (int): Number of predictive samples to generate and use to estimate likelihood.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Samples and log probabilities drawn from the joint distribution.
    """

    samples_list: List[torch.Tensor] = []
    sample_logprobs_list: List[torch.Tensor] = []

    # Expand tensors for efficient computation.
    xc_ = einops.repeat(xc, "m n d -> s m n d", s=num_samples)
    yc_ = einops.repeat(yc, "m n d -> s m n d", s=num_samples)
    xt_ = einops.repeat(xt, "m n d -> s m n d", s=num_samples)
    xc_, _ = compress_batch_dimensions(xc_, other_dims=2)
    yc_, _ = compress_batch_dimensions(yc_, other_dims=2)
    xt_, _ = compress_batch_dimensions(xt_, other_dims=2)

    # AR mode for loop.
    for i in range(xt_.shape[1]):
        with torch.no_grad():
            # Compute conditional distribution, sample and evaluate log probabilities.
            pred_dist = model(xc=xc_, yc=yc_, xt=xt_[:, i : i + 1])
            pred = pred_dist.rsample()
            pred_logprob = pred_dist.log_prob(pred)

            # Store samples and probabilities.
            pred = pred.detach()
            pred_logprob = pred_logprob.detach()
            samples_list.append(pred)
            sample_logprobs_list.append(pred_logprob)

            # Update context.
            xc_ = torch.cat([xc_, xt_[:, i : i + 1]], dim=1)
            yc_ = torch.cat([yc_, pred], dim=1)

    # Compute log probability of sample.
    samples = torch.cat(samples_list, dim=1)
    sample_logprobs = torch.cat(sample_logprobs_list, dim=1)

    samples = einops.rearrange(samples, "(s m) n d -> s m n d", s=num_samples)
    sample_logprobs = einops.rearrange(
        sample_logprobs, "(s m) n d -> s m n d", s=num_samples
    )
    sample_logprobs = sample_logprobs.mean(0)

    return samples, sample_logprobs


def updated_ar_predict(
    model: nn.Module,
    xc: torch.Tensor,
    yc: torch.Tensor,
    xt: torch.Tensor,
    tidx_blocks: torch.Tensor,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    samples_list: List[torch.Tensor] = []
    sample_logprobs_list: List[torch.Tensor] = []

    # Expand tensors for efficient computation.
    xc_ = einops.repeat(xc, "m n d -> s m n d", s=num_samples)
    yc_ = einops.repeat(yc, "m n d -> s m n d", s=num_samples)
    xt_ = einops.repeat(xt, "m n d -> s m n d", s=num_samples)
    xc_, _ = compress_batch_dimensions(xc_, other_dims=2)
    yc_, _ = compress_batch_dimensions(yc_, other_dims=2)
    xt_, _ = compress_batch_dimensions(xt_, other_dims=2)

    # Clear xc_cache.
    model.clear_cache()

    # AR mode for loop.
    for tidx_block in tidx_blocks:
        with torch.no_grad():
            # Compute conditional distribution, sample and evaluate log probabilities.

            # Select block of targets to predict.
            xt_block = xt_[:, tidx_block, ...]

            # Get pred distribution for this block of targets.
            pred_dist = model.ar_forward(xc=xc_, yc=yc_, xt=xt_block)
            pred = pred_dist.sample()
            pred_logprob = pred_dist.log_prob(pred)

            # Store samples and probabilities.
            samples_list.append(pred)
            sample_logprobs_list.append(pred_logprob)

            # Set xc_ and yc_ to xt_block and pred.
            xc_ = xt_block
            yc_ = pred

    # Clear cache again.
    model.clear_cache()

    # Compute log probability of sample.
    samples = torch.cat(samples_list, dim=1)
    sample_logprobs = torch.cat(sample_logprobs_list, dim=1)

    samples = einops.rearrange(samples, "(s m) n d -> s m n d", s=num_samples)
    sample_logprobs = einops.rearrange(
        sample_logprobs, "(s m) n d -> s m n d", s=num_samples
    )
    sample_logprobs = sample_logprobs.mean(0)

    return samples, sample_logprobs