import dataclasses
from typing import Any, Callable, List, Optional

import lightning.pytorch as pl
import torch
from torch import nn

from ..data.base import Batch
from .experiment_utils import ModelCheckpointer, np_loss_fn, np_pred_fn, discrete_denoising_loglik
from ..schedulers.base import BaseScheduler


class LitWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        scheduler: Optional[BaseScheduler] = None,
        optimiser: Optional[torch.optim.Optimizer] = None,
        loss_fn: Callable = np_loss_fn,
        pred_fn: Callable = np_pred_fn,
        loglik_fn: Callable = np_loss_fn,
        plot_fn: Optional[Callable] = None,
        checkpointer: Optional[ModelCheckpointer] = None,
        plot_interval: int = 1,
        subsample_targets: bool = False,
        num_samples: int = 1,
        split_batch: bool = False,
        subsample_test_targets: bool = False,
    ):
        super().__init__()

        self.model = model
        self.scheduler = scheduler
        self.optimiser = (
            optimiser if optimiser is not None else torch.optim.Adam(model.parameters())
        )
        self.loss_fn = loss_fn
        self.pred_fn = pred_fn
        self.loglik_fn = loglik_fn
        self.plot_fn = plot_fn
        self.checkpointer = checkpointer
        self.plot_interval = plot_interval
        self.subsample_targets = subsample_targets
        self.num_samples = num_samples
        self.split_batch = split_batch
        self.subsample_test_targets = subsample_test_targets
        self.val_outputs: List[Any] = []
        self.test_outputs: List[Any] = []
        self.train_losses: List[Any] = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> torch.Tensor:
        _ = batch_idx
        if self.scheduler is not None:
            loss = self.loss_fn(
                model=self.model, 
                batch=batch, 
                scheduler=self.scheduler, 
                subsample_targets=self.subsample_targets,
                )
        else:
            loss = self.loss_fn(self.model, batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.detach().cpu())
        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> None:
        _ = batch_idx
        result = {"batch": batch}
        # pred_dist = self.pred_fn(self.model, batch)
        # loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
        if self.scheduler is not None:
            loglik = -self.loss_fn(model=self.model, batch=batch, scheduler=self.scheduler, subsample_targets=self.subsample_targets)
        else:
            loglik = -self.loss_fn(self.model, batch)
        result["loglik"] = loglik.cpu()

        if hasattr(batch, "gt_pred") and batch.gt_pred is not None:
            _, _, gt_loglik, gt_loglik_joint = batch.gt_pred(
                xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
            )
            gt_loglik = gt_loglik.sum() / batch.yt[..., 0].numel()
            result["gt_loglik"] = gt_loglik.cpu()

        self.val_outputs.append(result)

    def test_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> None:
        _ = batch_idx
        result = {"batch": _batch_to_cpu(batch)}
        # pred_dist = self.pred_fn(self.model, batch)
        # loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
        loglik, loglik_joint = self.loglik_fn(
            model=self.model,
            batch=batch,
            scheduler=self.scheduler,
            num_samples=self.num_samples,
            split_batch=self.split_batch,
            subsample_targets=self.subsample_test_targets,
        )
        result["loglik"] = loglik.cpu()
        if loglik_joint is not None:
            result["loglik_joint"] = loglik_joint.cpu()

        if hasattr(batch, "gt_pred") and batch.gt_pred is not None:
            _, _, gt_loglik, gt_loglik_joint = batch.gt_pred(
                xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
            )
            gt_loglik = gt_loglik.sum() / batch.yt[..., 0].numel()
            gt_loglik_joint = gt_loglik_joint.sum() / batch.yt[..., 0].numel()

            result["gt_loglik"] = gt_loglik.cpu()
            result["gt_loglik_joint"] = gt_loglik_joint.cpu()

        self.test_outputs.append(result)

    def on_train_epoch_end(self) -> None:
        train_losses = torch.stack(self.train_losses)
        self.train_losses = []

        if self.checkpointer is not None:
            # For checkpointing.
            train_result = {
                "mean_loss": train_losses.mean(),
                "std_loss": train_losses.std() / (len(train_losses) ** 0.5),
            }
            self.checkpointer.update_best_and_last_checkpoint(
                model=self.model,
                val_result=train_result,
                prefix="train_",
                update_last=False,
            )

    def on_validation_epoch_end(self) -> None:
        results = {
            k: [result[k] for result in self.val_outputs]
            for k in self.val_outputs[0].keys()
        }
        self.val_outputs = []

        loglik = torch.stack(results["loglik"])
        mean_loglik = loglik.mean()
        std_loglik = loglik.std() / (len(loglik) ** 0.5)
        self.log("val/loglik", mean_loglik)
        self.log("val/std_loglik", std_loglik)

        if self.checkpointer is not None:
            # For checkpointing.
            val_result = {
                "mean_loss": -mean_loglik,
                "std_loss": std_loglik,
            }
            self.checkpointer.update_best_and_last_checkpoint(
                model=self.model, val_result=val_result, prefix="val_"
            )

        if "gt_loglik" in results:
            gt_loglik = torch.stack(results["gt_loglik"])
            mean_gt_loglik = gt_loglik.mean()
            std_gt_loglik = gt_loglik.std() / (len(gt_loglik) ** 0.5)
            self.log("val/gt_loglik", mean_gt_loglik)
            self.log("val/std_gt_loglik", std_gt_loglik)

        if self.plot_fn is not None and self.current_epoch % self.plot_interval == 0:
            self.plot_fn(
                model=self.model, 
                batches=results["batch"], 
                name=f"epoch-{self.current_epoch:04d}",
                scheduler=self.scheduler,
            )

    def configure_optimizers(self):
        return self.optimiser


def _batch_to_cpu(batch: Batch):
    batch_kwargs = {
        field.name: (
            getattr(batch, field.name).cpu()
            if isinstance(getattr(batch, field.name), torch.Tensor)
            else getattr(batch, field.name)
        )
        for field in dataclasses.fields(batch)
    }
    return type(batch)(**batch_kwargs)
