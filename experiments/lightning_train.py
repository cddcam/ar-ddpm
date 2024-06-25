import lightning.pytorch as pl
from plot import plot
from plot_image import plot_image

from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.lightning_utils import LitWrapper



def main():
    experiment, checkpointer = initialize_experiment()

    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    epochs = experiment.params.epochs
    scheduler = experiment.scheduler

    def plot_fn(model, batches, name, scheduler):
        if experiment.params.dim_x == 1:
            plot(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                plot_ar_mode=experiment.misc.plot_ar_mode,
                num_ar_samples=20,
                name=name,
                pred_fn=experiment.misc.pred_fn,
                scheduler=scheduler,
                subsample_targets=experiment.misc.subsample_targets,
            )
        else:
            plot_image(
                model=model, 
                batches=batches,
                num_fig=min(5, len(batches)),
                name=name,
                pred_fn=experiment.misc.pred_fn,
                scheduler=scheduler,
                subsample_targets=experiment.misc.subsample_targets,
            )

    lit_model = LitWrapper(
        model=model,
        scheduler=scheduler,
        optimiser=optimiser,
        loss_fn=experiment.misc.loss_fn,
        pred_fn=experiment.misc.pred_fn,
        plot_fn=plot_fn,
        checkpointer=checkpointer,
        plot_interval=experiment.misc.plot_interval,
        subsample_targets=experiment.misc.subsample_targets,
        num_samples=experiment.misc.num_loglik_samples,
        split_batch=experiment.misc.split_batch,
        subsample_test_targets=experiment.misc.subsample_test_targets,
    )
    logger = pl.loggers.WandbLogger() if experiment.misc.logging else False
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        limit_train_batches=gen_train.num_batches,
        limit_val_batches=gen_val.num_batches,
        log_every_n_steps=1,
        devices=1,
        gradient_clip_val=experiment.misc.gradient_clip_val,
    )

    trainer.fit(model=lit_model, train_dataloaders=gen_train, val_dataloaders=gen_val)


if __name__ == "__main__":
    main()
