import os
import json
import shutil
import torch
import h5py

def save_config_to_json(path, config):
    """Saves config to json file
    """
    with open(os.path.join(path, 'config.json'), 'w') as file:
        json.dump(config, file)


def save_checkpoint(state, is_best, checkpoint, is_final=False):
    """Saves model and training parameters
    at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
       state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
       is_best: (bool) True if it is the best model seen till now
       checkpoint: (string) folder where parameters are to be saved
    """
    if not os.path.exists(checkpoint):
       print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
       os.mkdir(checkpoint)

    if is_final:
        torch.save(state, os.path.join(checkpoint, 'final.pth.tar'))
    else:
        filepath = os.path.join(checkpoint, 'last.pth.tar')
        torch.save(state, filepath)
        if is_best:
           shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def save_grid(state, checkpoint, filename):

    if not os.path.exists(checkpoint):
       print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
       os.mkdir(checkpoint)
    torch.save(state, os.path.join(checkpoint, filename))


def save_scene(tsdf, occ, latent, conf=None, scene_id=None, path=None, mode='train'):
    if not os.path.exists(os.path.join(path, 'logs', 'output', 'last', mode)):
        os.makedirs(os.path.join(path, 'logs', 'output', 'last', mode))

    tsdf_file = os.path.join(path, 'logs', 'output', 'last', mode,
                             '{}.tsdf.hf5'.format(scene_id.replace('/', '.')))
    latent_file = os.path.join(path, 'logs', 'output', 'last', mode,
                               '{}.latent.hf5'.format(
                                   scene_id.replace('/', '.')))
    occ_file = os.path.join(path, 'logs', 'output', 'last', mode,
                            '{}.occ.hf5'.format(
                                   scene_id.replace('/', '.')))

    conf_file = os.path.join(path, 'logs', 'output', 'last', mode,
                            '{}.conf.hf5'.format(
                                scene_id.replace('/', '.')))

    # save results
    with h5py.File(tsdf_file, 'w') as hf:
        hf.create_dataset("TSDF",
                          shape=tsdf.shape,
                          data=tsdf)

    # save results
    with h5py.File(occ_file, 'w') as hf:
        hf.create_dataset("occupancy",
                          shape=occ.shape,
                          data=occ)

    with h5py.File(latent_file, 'w') as hf:
        hf.create_dataset("latent",
                          shape=latent.shape,
                          data=latent)

    if conf is not None:
        with h5py.File(conf_file, 'w') as hf:
            hf.create_dataset("conf",
                              shape=conf.shape,
                              data=conf)

def save_config_to_lightning(logger, config):

    path = os.path.join(logger.save_dir, logger.name, 'version_{}'.format(logger.version))

    with open(os.path.join(path, 'config.json'), 'w') as file:
        json.dump(config, file)

def save_scene_to_lightning(logger, data, scene_id,  mode='train'):

    path = os.path.join(logger.save_dir, logger.name, 'version_{}'.format(logger.version))

    tsdf = data['tsdf']
    occ = data['occ']
    latent = data['latent']

    try:
        conf = data['conf']
    except KeyError:
        conf = None


    if not os.path.exists(os.path.join(path, 'scenes', 'last', mode)):
        os.makedirs(os.path.join(path, 'scenes', 'last', mode))

    tsdf_file = os.path.join(path, 'scenes', 'last', mode,
                             '{}.tsdf.hf5'.format(scene_id.replace('/', '.')))
    latent_file = os.path.join(path, 'scenes', 'last', mode,
                               '{}.latent.hf5'.format(
                                   scene_id.replace('/', '.')))
    occ_file = os.path.join(path, 'scenes', 'last', mode,
                            '{}.occ.hf5'.format(
                                   scene_id.replace('/', '.')))

    conf_file = os.path.join(path, 'scenes', 'last', mode,
                            '{}.conf.hf5'.format(
                                scene_id.replace('/', '.')))

    # save results
    with h5py.File(tsdf_file, 'w') as hf:
        hf.create_dataset("TSDF",
                          shape=tsdf.shape,
                          data=tsdf)

    # save results
    with h5py.File(occ_file, 'w') as hf:
        hf.create_dataset("occupancy",
                          shape=occ.shape,
                          data=occ)

    with h5py.File(latent_file, 'w') as hf:
        hf.create_dataset("latent",
                          shape=latent.shape,
                          data=latent)

    if conf is not None:
        with h5py.File(conf_file, 'w') as hf:
            hf.create_dataset("conf",
                              shape=conf.shape,
                              data=conf)

