import copy
import os
from typing import Callable, List, Tuple

import einops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.distributions as td

import wandb
from tnp.data.image import GriddedImageBatch, ImageBatch
from tnp.utils.experiment_utils import np_pred_fn
from tnp.schedulers.base import BaseScheduler

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_image(
    model: nn.Module,
    scheduler: BaseScheduler,
    batches: List[ImageBatch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (24.0, 8.0),
    name: str = "plot",
    subplots: bool = True,
    savefig: bool = False,
    logging: bool = True,
    pred_fn: Callable = np_pred_fn,
    num_np_samples: int = 16,
    test_sampling: bool = False,
    subsample_targets: bool = False,
):
    for i in range(num_fig):
        batch = batches[i]

        x = batch.x[:1]
        y = batch.y[:1]
        xc = batch.xc[:1]
        yc = batch.yc[:1]
        xt = batch.xt[:1]
        yt = batch.yt[:1]
        mc = batch.mc[:1]

        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = yt

        if isinstance(batch, GriddedImageBatch):
            mc_grid = batch.mc_grid[:1]
            y_grid = batch.y_grid[:1]
            mt_grid = batch.mt_grid[:1]
            batch.mc_grid = mc_grid
            batch.y_grid = y_grid
            batch.mt_grid = mt_grid

        plot_batch = copy.deepcopy(batch)
        plot_batch.xt = x

        if isinstance(batch, GriddedImageBatch):
            assert isinstance(batch, GriddedImageBatch)
            plot_batch.mt_grid = torch.full(batch.mt_grid.shape, True)

        with torch.no_grad():
            
            yt_pred_dist, y_plot_pred_dist, noised_targets, tt = pred_fn(
                model=model, 
                batch=batch,
                scheduler=scheduler,
                x_plot=plot_batch.xt,   
                subsample_targets=subsample_targets)
            
            tt = tt[0].cpu().numpy()
            
            if isinstance(y_plot_pred_dist, td.Normal):
                mean, std = (
                    y_plot_pred_dist.mean.cpu().numpy(),
                    y_plot_pred_dist.stddev.cpu().numpy(),
                )
            else:
                mean, std = (
                    y_plot_pred_dist.mean.cpu().numpy()[..., None],
                    y_plot_pred_dist.stddev.cpu().numpy()[..., None],
                )
            # model_nll = -yt_pred_dist.log_prob(yt).sum() / batch.yt[..., 0].numel()
        # Reorganise into grid.
        if y.shape[-1] == 1:
            mc_ = einops.repeat(mc[:1], "m n -> m n d", d=y.shape[-1])
            yc_ = np.ma.masked_where(
                ~mc_.cpu().numpy(),
                y[:1, :].cpu().numpy(),
            )
        else:
            # Masking does not work for RGB images.
            # Use mask to control alpha values instead.
            mc_ = einops.rearrange(mc[:1], "m n -> m n 1")
            yc_ = torch.cat((y[:1], mc_), dim=-1).cpu().numpy()

        w = int(yc_.shape[-2] ** 0.5)
        yc_ = einops.rearrange(yc_, "1 (n m) d -> n m d", n=w, m=w)
        y_ = einops.rearrange(y[:1, :].cpu().numpy(), "1 (n m) d -> n m d", n=w, m=w)
        mean = einops.rearrange(mean, "1 (n m) d -> n m d", n=w, m=w)
        std = einops.rearrange(std, "1 (n m) d -> n m d", n=w, m=w)

        if subplots:
            # Make figure for plotting
            fig, axes = plt.subplots(figsize=figsize, ncols=3, nrows=1)

            axes[0].imshow(yc_, cmap="gray", vmax=1, vmin=0)
            axes[1].imshow(mean, cmap="gray", vmax=1, vmin=0)
            axes[2].imshow(std, cmap="gray", vmax=std.max(), vmin=std.min())

            axes[0].set_title("Context set", fontsize=18)
            axes[1].set_title(f"Mean prediction, time_layer={tt}", fontsize=18)
            axes[2].set_title("Std prediction", fontsize=18)

            plt.suptitle(
                f"prop_ctx = {xc.shape[-2] / x.shape[-2]:.2f}",
                #
                #f"NLL = {model_nll:.3f}",
                fontsize=24,
            )

            fname = f"fig/{name}/{i:03d}"
            if wandb.run is not None and logging:
                wandb.log({fname: wandb.Image(fig)})
            elif savefig:
                if not os.path.isdir(f"fig/{name}"):
                    os.makedirs(f"fig/{name}")
                plt.savefig(fname)
            else:
                plt.show()

            plt.close()

        else:
            for fig_name, y_plot in zip(
                ("context", "ground_truth", "pred_mean"), (yc_, y_, mean)
            ):
                fig = plt.figure(figsize=figsize)

                plt.imshow(y_plot, vmax=1, vmin=0)
                plt.tight_layout()

                fname = f"fig/{name}/{i:03d}/{fig_name}"
                if wandb.run is not None and logging:
                    wandb.log({fname: wandb.Image(fig)})
                elif savefig:
                    if not os.path.isdir(f"fig/{name}/{i:03d}"):
                        os.makedirs(f"fig/{name}/{i:03d}")
                    plt.savefig(fname)
                else:
                    plt.show()

                plt.close()
