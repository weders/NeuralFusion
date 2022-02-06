import pytorch_lightning as pl
import os
import shutil
import glob
import json

from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from copy import deepcopy

from utils.saving import save_config_to_lightning

class IterativeTestingCallback(pl.Callback):

    def __init__(self, seed):

        super().__init__()
        self.seed = seed

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):

        scene_id = batch['scene_id'][0]

        results = pl_module._evaluate_database(pl_module._test_database,
                                               mode='test',
                                               scene=scene_id,
                                               save=False)

        logger = pl_module.logger

        save_dir = logger.save_dir
        name = logger.name
        version = logger.version

        results_path = os.path.join(save_dir, name, 'version_{}'.format(version), 'results')

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        iterative_file = os.path.join(results_path, 'iterative_{}.json'.format(self.seed))

        # load json
        if os.path.exists(iterative_file):
            with open(iterative_file, 'r') as file:
                data = json.load(file)

            if scene_id in data:
                for k in results['log'].keys():
                    data[scene_id][k] += [results['log'][k]]
            else:
                data[scene_id] = {}
                for k in results['log'].keys():
                    data[scene_id][k] = [results['log'][k]]

        else:
            data = dict()
            data[scene_id] = dict()

            for k in results['log'].keys():
                data[scene_id][k] = [results['log'][k]]

        # save json
        with open(iterative_file, 'w') as file:
            json.dump(data, file)

        return results

class CustomModelCheckpoint(ModelCheckpoint):

    def __init__(self,
                 filepath=None,
                 monitor='val_loss',
                 verbose=False,
                 save_last=False,
                 save_top_k=1,
                 save_weights_only=False,
                 mode='auto',
                 period=1,
                 prefix=''):

        super().__init__()


    def on_save_checkpoint(self, trainer, pl_module):
        return super().on_save_checkpoint(self, trainer, pl_module)

    def save_checkpoint(self, trainer, pl_module):
        super().save_checkpoint(trainer, pl_module)


class ReconstructionCheckpoint(pl.Callback):

    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        logger = pl_module.logger

        save_dir = logger.save_dir
        name = logger.name
        version = logger.version

        checkpoints_path = os.path.join(save_dir, name, 'version_{}'.format(version), 'checkpoints')

        checkpoint_files = glob.glob(os.path.join(checkpoints_path, 'epoch=*.ckpt'))

        if len(checkpoint_files) > 0:
            best_epoch_file = checkpoint_files[-1]
            best_epoch_file = best_epoch_file.split('/')[-1]

            best_epoch = best_epoch_file.split('=')[1].replace('-step', '')
            best_epoch = int(best_epoch)

            reconstruction_path = os.path.join(save_dir, name, 'version_{}'.format(version), 'scenes')

            reconstruction_path_last = os.path.join(reconstruction_path, 'last')
            reconstruction_path_best = os.path.join(reconstruction_path, 'best')

            if trainer.current_epoch - 1 == best_epoch:
                shutil.rmtree(reconstruction_path_best,
                            ignore_errors=True)
                shutil.copytree(reconstruction_path_last,
                                reconstruction_path_best)

class DatabaseCallback(pl.Callback):

    def __init__(self, config):

        super().__init__()

        self.config = config

    def on_train_epoch_start(self, trainer, pl_module):

        pl_module._train_database.reset()
        pl_module._val_database.reset()


class ConfigLoggingCallback(pl.Callback):

    def __init__(self):

        super().__init__()

    def on_train_start(self, trainer, pl_module):

        config_to_save = deepcopy(pl_module.config)
        config_to_save.PIPELINE.device = None

        save_config_to_lightning(pl_module.logger,
                                 config_to_save)

        print('Saved config to logs ...')
