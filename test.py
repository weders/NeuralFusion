import pytorch_lightning as pl
import os
import argparse
import platform
import random

from pytorch_lightning.loggers import TensorBoardLogger

from training.pipeline import NeuralFusionPipeline
from utils.loading import load_config_from_json, load_config_from_yaml
from training.utils import *
from training.callbacks import *

from pytorch_lightning import seed_everything

def arg_parser():

    parser = argparse.ArgumentParser()

    # add arguments to parser
    parser.add_argument('--test')
    
    parser.add_argument('--root_path')
    parser.add_argument('--data_path', dest='data_path')

    parser.add_argument('--experiment', dest='experiment')
    parser.add_argument('--version', dest='version')
    parser.add_argument('--checkpoint', dest='checkpoint')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    return args

def main(args):

    seed_everything(args.seed)

    # define all the paths and files
    experiment_path = os.path.join(args.root_path, args.experiment, 'version_{}'.format(args.version))

    # all files needed
    checkpoint_file = os.path.join(experiment_path, 'checkpoints', args.checkpoint)
    config_file = os.path.join(experiment_path, 'config.json')
    hparams_file = os.path.join(experiment_path, 'hparams.yaml')

    # load config
    config = load_config_from_json(experiment_path)

    # load config
    test_config = load_config_from_yaml(args.test)

    test_config.DATA.root_dir = args.data_path
    test_config.SETTINGS.data_path = args.data_path

    # load pipeline
    pipeline = NeuralFusionPipeline.load_from_checkpoint(checkpoint_path=checkpoint_file,
                                                         hparams_file=hparams_file,
                                                         strict=True)    

    pipeline.config.DATA = test_config.DATA
    pipeline.config.SETTINGS.data_path = args.data_path

    logger = TensorBoardLogger(save_dir=args.root_path,
                               name=args.experiment,
                               version=args.version,
                               filename_suffix='test')

    # intialize trainer
    trainer = pl.Trainer(gpus=1,
                         logger=logger)


    test_data = get_dataset(test_config, mode='test')
    test_loader = get_data_loader(test_data, test_config.TESTING)

    # test pipeline
    trainer.test(pipeline, test_loader)


if __name__ == '__main__':

    args = arg_parser()
    main(args)
