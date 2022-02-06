import torch
import argparse
import pytorch_lightning as pl

from pytorch_lightning import loggers
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint

from training.pipeline import NeuralFusionPipeline
from training.callbacks import DatabaseCallback, ConfigLoggingCallback, ReconstructionCheckpoint

from training.utils import *
from utils.saving import *
from utils.loading import *


def arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', dest='config')
    parser.add_argument('--data_path', dest='data_path')
    parser.add_argument('--experiment_path', dest='experiment_path')
    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--no_eval', action='store_true', default=False)

    args = parser.parse_args()

    return args

def main(args):


    config = load_config_from_yaml(args.config)
    
    config.SETTINGS.data_path = args.data_path
    config.SETTINGS.experiment_path = args.experiment_path

    # define datasets
    train_data = get_dataset(config, mode='train')
    val_data = get_dataset(config, mode='val')
    test_data = get_dataset(config, mode='test')

    # define data loaders
    train_loader = get_data_loader(train_data, config.TRAINING)
    val_loader = get_data_loader(val_data, config.VALIDATION)
    test_loader = get_data_loader(test_data, config.TESTING)

    # init tensorboard logging
    logger = loggers.TensorBoardLogger(save_dir=config.SETTINGS.experiment_path,
                                       name=config.SETTINGS.name)
    
    # enable anomaly detection for better debugging
    torch.autograd.set_detect_anomaly(True)

    # init training pipeline
    pipeline = NeuralFusionPipeline(config=config)

    # define all callbacks
    database_callback = DatabaseCallback(config)
    config_logger_callback = ConfigLoggingCallback()
    checkpoint_callback = ModelCheckpoint(verbose=True, monitor='val/f1', save_last=True, mode='max')
    reconstruction_callback = ReconstructionCheckpoint()


    trainer = pl.Trainer(callbacks=[database_callback, config_logger_callback, RichProgressBar(), checkpoint_callback, reconstruction_callback], 
                         track_grad_norm=2,
                         gpus=1,
                         logger=logger,
                         gradient_clip_val=1.,
                         num_sanity_val_steps=0,
                         max_epochs=args.n_epochs,
                         accumulate_grad_batches=config.TRAINING.accumulate_steps)
                         
    trainer.fit(pipeline, train_loader, val_loader)

    if not args.no_eval:
        # test trained pipeline
        trainer.test(pipeline, test_loader)


if __name__ == '__main__':
    args = arg_parser()
    main(args)