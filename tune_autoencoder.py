from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import pytorch_lightning
from lib.correspondence_datamodule import CorrespondenceDataModule

from lib.autoencoder import FeatureEncoder
from pytorch_lightning.loggers import TensorBoardLogger

def train_autoencoder_tune(config, num_epochs=10):
    autoencoder = FeatureEncoder(config_=config)

    tb_logger = TensorBoardLogger('tb_logs', name='autoencoder')
    trainer = pytorch_lightning.Trainer(
            logger=tb_logger,
            max_epochs=num_epochs,
            callbacks=[
                TuneReportCallback(
                    {
                        "loss": "ptl/val_loss",
                    },
                    on="validation_end")
            ],
            gpus=1 if torch.cuda.is_available() else None)
    dm = CorrespondenceDataModule()
    trainer.fit(autoencoder, dm)


if __name__ == "__main__":
    num_samples = 10
    num_epochs = 10
    config = {
            "embedded_layers": tune.choice([32, 64, 128]),
            "lr": tune.loguniform(1e-4, 1e-2),
    }

    reporter = tune.CLIReporter(
        parameter_columns=["encoded_channels", "lr"],
        metric_columns=["loss", "training_iteration"]
    )

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    analysis = tune.run(
        tune.with_parameters(
            train_autoencoder_tune,
            num_epochs=num_epochs,
        ),
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_asha"
    )

    print("Best hyperparameters found were: ", analysis.best_config)
