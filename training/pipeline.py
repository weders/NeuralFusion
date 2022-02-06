import torch
import time
import random
import pytorch_lightning as pl

from easydict import EasyDict
from pipeline.pipeline import FusionPipeline
from training.database import Database

from .utils import *
from utils.saving import *
from utils.metrics import *


class NeuralFusionPipeline(pl.LightningModule):
    
    def __init__(self, config):

        super().__init__()

        self.save_hyperparameters()
        self.config = config

        self._pipeline = FusionPipeline(self.config.PIPELINE)

        self._train_database = None
        self._val_database = None
        self._test_database = None

        self.criterion3 = torch.nn.BCELoss(reduction='none')

    def setup(self, stage):
       
        if stage == 'fit':
            self._train_database = Database(self.train_dataloader().dataset, self.config.DATABASE)
            self._val_database = Database(self.val_dataloader().dataset, self.config.DATABASE)
            self._test_database = Database(self.test_dataloader().dataset, self.config.DATABASE)

        elif stage == 'test':
            self._test_database = Database(self.test_dataloader().dataset, self.config.DATABASE)

    def train_dataloader(self):
        train_data = get_dataset(self.config, mode='train')
        train_loader = get_data_loader(train_data, self.config.TRAINING)
        return train_loader

    def val_dataloader(self):
        val_data = get_dataset(self.config, mode='val')
        val_loader = get_data_loader(val_data, self.config.VALIDATION)
        return val_loader

    def test_dataloader(self):
        test_data = get_dataset(self.config, mode='test')
        test_loader = get_data_loader(test_data, self.config.TESTING)
        return test_loader

    def training_step(self, batch, batch_idx):

        # get scene id
        scene_id = batch['scene_id'][0]

        # get device
        device = batch[self.config.DATA.key].get_device()

        # query database
        batch['volume'] = torch.Tensor(self._train_database[scene_id]['latent'].data).to(device)
        batch['origin'] = torch.Tensor(self._train_database[scene_id]['latent'].origin).to(device)
        batch['count'] = torch.Tensor(self._train_database[scene_id]['count'])
        batch['extrinsics'] = batch['extrinsics']
        batch['intrinsics'] = batch['intrinsics']
        batch['input'] = batch[self.config.DATA.key]
        batch['original_mask'] = batch['original_mask']
        batch['resolution'] = self._train_database[scene_id]['latent'].resolution
        batch['gt'] = self._train_database[scene_id]['gt']

        if self.config.LOSS.reinitialize:
            p = random.uniform(0, 1)
            if p < self.config.LOSS.ratio:
                batch['volume'] = torch.zeros_like(batch['volume'])
                batch['count'] = torch.zeros_like(batch['count'])

        groundtruth_grid = torch.Tensor(batch['gt'].volume).unsqueeze_(0).clone()
        old_latent_grid = batch['volume'].clone()

        if self.config.LOSS.clamp:
            groundtruth_grid[groundtruth_grid < self.config.LOSS.min_clamp] = self.config.LOSS.max_clamp
            groundtruth_grid = groundtruth_grid.clamp(self.config.LOSS.min_clamp, self.config.LOSS.max_clamp)

        # fusion in latent space
        updated_grid, updates, count = self._pipeline.forward(batch)

        torch.cuda.empty_cache()
        # compute loss mask
        if self.config.LOSS.mask == 'updated':
            old_latent_grid = old_latent_grid.cpu().detach().clone().squeeze_(0).permute(-1, 0, 1, 2).unsqueeze_(0)
            mask, orig_mask = get_loss_mask(updated_grid.cpu().detach(), self.config.LOSS.mask_dilation_iterations, mode='updated', old_state=old_latent_grid, return_orig=True)        
        else:
            old_latent_grid = old_latent_grid.cpu().detach().clone().squeeze_(0).permute(-1, 0, 1, 2).unsqueeze_(0)
            mask, orig_mask = get_loss_mask(updated_grid.cpu().detach(), self.config.LOSS.mask_dilation_iterations, old_state=old_latent_grid, return_orig=True)

        mask = mask.to(device)

        if self.config.TRAINING.latent_noise:
            orig_mask = orig_mask.to(device)
            noise = 0.01 * torch.randn_like(updated_grid)
            noise[:, :, orig_mask == 0] = 0
            updated_grid[:, :, orig_mask == 1] += noise[:, :, orig_mask == 1]

        points_to_translate, values_gt = get_training_points(self.config, mask.cpu(), groundtruth_grid)
        points_to_translate = points_to_translate.to(device)

        # translate points
        values = self._pipeline.translate(points_to_translate, updated_grid)

        torch.cuda.empty_cache()

        values_gt = values_gt.to(device)

        mask = mask.unsqueeze_(0)

        # predictions
        values_sdf_est = values[:, 0]
        values_cls_est = values[:, 1]

        # groundtruth
        values_cls_gt = values_gt.clone()
        values_cls_gt[values_cls_gt >= 0.] = 0.
        values_cls_gt[values_cls_gt < 0.] = 1.

        # loss functions
        loss_occ = torch.mean(self.criterion3.forward(values_cls_est, values_cls_gt))
        loss_sdf_l1 = torch.mean(torch.abs(values_sdf_est - values_gt))
        loss_sdf_l2 = torch.mean(torch.pow(values_sdf_est - values_gt, 2))

         # compute overall loss function
        loss = self.config.LOSS.wl1 * loss_sdf_l1 + self.config.LOSS.wl2 * loss_sdf_l2 + self.config.LOSS.wlocc * loss_occ
        loss = self.config.LOSS.scale * loss
        
        # regularization
        if self.config.LOSS.latent_reg:

            b, n, h, w, d = updated_grid.shape

            updated_grid = updated_grid.view(1, self.config.DATABASE.n_features, h*w*d)
            if self.config.PIPELINE.use_count:
                latent_regularization = torch.mean(torch.var(updated_grid[:, :-1, :], dim=-1))
            else:
                latent_regularization = torch.mean(torch.var(updated_grid, dim=-1))
            updated_grid = updated_grid.view(1, self.config.DATABASE.n_features, h, w, d)
            loss += self.config.LOSS.wlreg * latent_regularization


        # bring feature grid into numpy shape and update database
        updated_grid_np = updated_grid.cpu().clone().permute(0, 2, 3, 4, 1)
        updated_grid_np = updated_grid_np.squeeze_(0)
        self._train_database.update(scene_id, updated_grid_np.cpu().detach().numpy().copy(), count.cpu().detach().numpy())

        logs = {'train_loss' : loss}

        # pack outputs
        output = {'loss' : loss, 'log' : logs}

        del updated_grid, updates, count, groundtruth_grid

        return output

    def test_step(self, batch, batch_idx):

        scene_id = batch['scene_id'][0]

        # get device
        device = batch[self.config.DATA.key].get_device()
        self.database_device = device

        # query database
        batch['volume'] = torch.Tensor(self._test_database[scene_id]['latent'].data).to(batch[self.config.DATA.key])
        batch['origin'] = torch.Tensor(self._test_database[scene_id]['latent'].origin).to(batch[self.config.DATA.key])
        batch['count'] = torch.Tensor(self._test_database[scene_id]['count'])
        batch['extrinsics'] = batch['extrinsics']
        batch['intrinsics'] = batch['intrinsics']
        batch['input'] = batch[self.config.DATA.key]
        batch['original_mask'] = batch['original_mask']
        batch['resolution'] = self._test_database[scene_id]['latent'].resolution
        batch['gt'] = self._test_database[scene_id]['gt']

        updated_grid, updates, count = self._pipeline.forward(batch)

         # bring feature grid in numpy shape
        updated_grid_np = updated_grid.cpu().clone().permute(0, 2, 3, 4, 1)
        updated_grid_np = updated_grid_np.squeeze_(0)
        self._test_database.update(scene_id, updated_grid_np.cpu().detach().numpy().copy(), count.cpu().detach().numpy())

        return {}

    def validation_step(self, batch, batch_idx):

        self._pipeline.eval()
        self._pipeline.eval()

        scene_id = batch['scene_id'][0]

        # get device
        device = batch[self.config.DATA.key].get_device()
        self.database_device = device

        # query database
        batch['volume'] = torch.Tensor(self._val_database[scene_id]['latent'].data).to(device)
        batch['origin'] = torch.Tensor(self._val_database[scene_id]['latent'].origin).to(device)
        batch['count'] = torch.Tensor(self._val_database[scene_id]['count'])
        batch['extrinsics'] = batch['extrinsics']
        batch['intrinsics'] = batch['intrinsics']
        batch['input'] = batch[self.config.DATA.key]
        batch['original_mask'] = batch['original_mask']
        batch['resolution'] = self._val_database[scene_id]['latent'].resolution
        batch['gt'] = self._val_database[scene_id]['gt']

        updated_grid, updates, count = self._pipeline.forward(batch)

         # bring feature grid in numpy shape
        updated_grid_np = updated_grid.cpu().clone().permute(0, 2, 3, 4, 1)
        updated_grid_np = updated_grid_np.squeeze_(0)
        self._val_database.update(scene_id, updated_grid_np.cpu().detach().numpy().copy(), count.cpu().detach().numpy())


        return {}

    def _evaluate_database(self, database, mode='train', scene=None, save=True):

        self._pipeline.eval()

        metric_container = {'mae' : 0.,
                            'mse' : 0.,
                            'iou' : 0.,
                            'acc' : 0.,
                            'f1' : 0.}

        scenes = database.ids if scene is None else [scene]

        for scene_id in scenes:

            self.config.DATA.factor = int(self.config.DATA.grid_resolution / self.config.DATA.evaluation_resolution)

            # get latent representation grid
            latent_grid = torch.Tensor(database[scene_id]['latent'].data).to(self.database_device)
            latent_grid = latent_grid.unsqueeze_(0).permute(0, 4, 1, 2, 3)

            # get groundtruth grid
            groundtruth_grid = torch.Tensor(database[scene_id]['gt'].volume).to(self.database_device)

            # render output modalities from latent representation grid
            rendered_grid = get_translation(self._pipeline,
                                            latent_grid,
                                            self.config)

            # extract output modalities
            sdf_grid = rendered_grid[:, 0, :, :, :]
            occ_grid = rendered_grid[:, 1, :, :, :]
            conf_grid = rendered_grid[:, -1, :, :, :] # confidence

            # compute mask
            latent_grid = latent_grid.cpu()


            mask = get_translation_mask(latent_grid, self.config, groundtruth_grid)

            # masking and clamping
            # sensor cannot see inside object, treating occupied space as freespace    
            groundtruth_grid[groundtruth_grid < self.config.LOSS.min_clamp] = self.config.LOSS.max_clamp

            # don't evaluate where nothing has been integrated    
            groundtruth_grid[mask == 0] = self.config.LOSS.max_clamp
            sdf_grid[mask.unsqueeze_(0) == 0] = self.config.LOSS.max_clamp

            # only evaluate for values in truncation band
            groundtruth_grid = torch.clamp(groundtruth_grid,
                                           self.config.LOSS.min_clamp,
                                           self.config.LOSS.max_clamp)
            groundtruth_grid.unsqueeze_(0)

            groundtruth_grid = groundtruth_grid.cpu()
            sdf_grid = sdf_grid.cpu()
            mask = mask.cpu()

            # compute metrics
            metric_container['mae'] += mad_fn(sdf_grid, groundtruth_grid, mask)
            metric_container['mse'] += mse_fn(sdf_grid, groundtruth_grid, mask)
            metric_container['iou'] += iou_fn(sdf_grid, groundtruth_grid, mask)
            metric_container['acc'] += acc_fn(sdf_grid, groundtruth_grid, mask)
            metric_container['f1'] += f1_fn(sdf_grid, groundtruth_grid, mask).item()

            #save to logs
            if save:
                 # prepare data for saving
                data = {}
                data['tsdf'] = sdf_grid.detach().numpy()
                data['occ'] = occ_grid.cpu().detach().numpy()
                data['conf'] = conf_grid.cpu().detach().numpy()
                data['latent'] = latent_grid.cpu().detach().numpy()

                save_scene_to_lightning(self.logger, data, scene_id, mode)

        logs = {}

        for m in metric_container.keys():
            metric_container[m] /= len(scenes)
            logs['{}/{}'.format(mode, str(m))] = metric_container[m]

        results = logs

        return results

    def test_epoch_end(self, outputs):

        self._pipeline.eval()
        results = self._evaluate_database(self._test_database, mode='test')
        self.log_dict(results)
    
    def training_epoch_end(self, outputs):

        self._pipeline.eval()
        results = self._evaluate_database(self._train_database, mode='train')
        self.log_dict(results)

    def validation_epoch_end(self, outputs):

        self._pipeline.eval()
        results = self._evaluate_database(self._val_database, mode='val')
        self.log_dict(results, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params' : self._pipeline._translator.parameters()},
                                      {'params' : self._pipeline._fusion.parameters(),
                                       'lr' : self.config.OPTIMIZATION.pipeline.lr}],
                                      lr=self.config.OPTIMIZATION.renderer.lr,
                                      betas=(0.9, 0.999),
                                      eps=1e-08,
                                      weight_decay=self.config.OPTIMIZATION.pipeline.weight_decay,
                                      amsgrad=False)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=self.config.OPTIMIZATION.scheduler.gamma)

        return [optimizer], [scheduler]
