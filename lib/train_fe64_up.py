from lib.autoencoder import *

autoencoder = FeatureEncoder64Up()
tb_logger = TensorBoardLogger('tb_logs', name='feature_encoder64_up_deep_lr2e4')
trainer = pytorch_lightning.Trainer(logger = tb_logger,
    gpus=1 if torch.cuda.is_available() else None)
dm = CorrespondenceDataModule()
trainer.fit(autoencoder, dm)