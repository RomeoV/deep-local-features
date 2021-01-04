from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import pytorch_lightning
from lib.correspondence_datamodule import CorrespondenceDataModule
import torch
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

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
            "encoded_channels": tune.choice([16, 32, 64, 128]),
            "lr": tune.loguniform(1e-4, 1e-2),
    }

    reporter = tune.CLIReporter(
        parameter_columns=["encoded_channels", "lr"],
        metric_columns=["loss", "training_iteration"]
    )

    bohb_algo = TuneBOHB(max_concurrent=1)
    scheduler = HyperBandForBOHB(
      time_attr="training_iteration",
      max_t=30
    )

    analysis = tune.run(
        tune.with_parameters(
            train_autoencoder_tune
        ),
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=config,
        scheduler=scheduler,
        search_alg=bohb_algo,
        progress_reporter=reporter,
        name="tune_autoencoder_bohb",
        metric="loss",
        mode="min",
    )

    print("Best hyperparameters found were: ", analysis.best_config)
