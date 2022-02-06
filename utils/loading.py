import yaml
import json
import os
import torch

from easydict import EasyDict


def load_config_from_yaml(path):
    """
    Method to load the config file for
    neural network training
    :param path: yaml-filepath with configs stored
    :return: easydict containing config
    """
    c = yaml.load(open(path))
    config = EasyDict(c)

    return config


def load_config_from_json(path):

    file = os.path.join(path, 'config.json')
    with open(file, 'r') as file:
        data = json.load(file)
    config = EasyDict(data)
    return config


def load_config_from_experiment(path):
    return load_config_from_json(path)


def load_experiment(checkpoint, pipeline, renderer, optimizer):

    _ = load_renderer(checkpoint, renderer)
    _ = load_pipeline(checkpoint, pipeline)
    checkpoint = load_optimizer(checkpoint, optimizer)

    return checkpoint


def load_optimizer(checkpoint, optimizer):

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint



def load_checkpoint(checkpoint, model, optimizer=None, key='state_dict'):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))

    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

        cleaned_dict = {}
        for k in checkpoint['state_dict']:
            cleaned_dict[k.replace('module.', '')] = checkpoint[key][k]

        model.load_state_dict(cleaned_dict)

    except KeyError:
        print('loading model partly')

        pretrained_dict = {k: v for k, v in checkpoint[key].items() if k in model.state_dict()}

        model.state_dict().update(pretrained_dict)
        model.load_state_dict(model.state_dict())



    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def load_renderer(checkpoint,
                  renderer):

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))

    print('loading renderer')
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint,
                                map_location=torch.device('cuda:0'))
    else:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    renderer.load_state_dict(checkpoint['renderer_state_dict'])

    return checkpoint


def load_pipeline(checkpoint, pipeline):

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))

    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint, map_location=torch.device('cuda:0'))
        else:
            checkpoint = torch.load(checkpoint,
                                    map_location=torch.device('cpu'))

        pipeline.load_state_dict(checkpoint['pipeline_state_dict'])

    except:
        print('loading model partly')

        pipeline_pretrained_dict = {k: v for k, v in
                                   checkpoint['pipeline_state_dict'].items() if
                                   k in pipeline.state_dict()}
        pipeline.state_dict().update(pipeline_pretrained_dict)
        pipeline.load_state_dict(pipeline.state_dict())

    return checkpoint


def load_encoder(checkpoint,
                  encoder):

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))

    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

        encoder.load_state_dict(checkpoint['encoder_state_dict'])

    except:
        print('loading model partly')

        encoder_pretrained_dict = {k: v for k, v in checkpoint['encoder_state_dict'].items() if k in encoder.state_dict()}
        encoder.state_dict().update(encoder_pretrained_dict)
        encoder.load_state_dict(encoder.state_dict())

    return checkpoint



def load_checkpoint_pretraining(checkpoint,
                                encoder,
                                renderer,
                                noise_renderer,
                                render_optimizer=None,
                                encoder_optimizer=None):

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))

    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        renderer.load_state_dict(checkpoint['render_state_dict'])
        noise_renderer.load_state_dict(checkpoint['noise_render_state_dict'])

    except:
        print('loading model partly')

        encoder_pretrained_dict = {k: v for k, v in checkpoint['encoder_state_dict'].items() if k in encoder.state_dict()}
        renderer_pretrained_dict = {k: v for k, v in checkpoint['render_state_dict'].items() if k in renderer.state_dict()}
        noise_renderer_pretrained_dict = {k: v for k, v in checkpoint['noise_render_state_dict'].items() if k in noise_renderer.state_dict()}

        encoder.state_dict().update(encoder_pretrained_dict)
        renderer.state_dict().update(renderer_pretrained_dict)
        noise_renderer.state_dict().update(noise_renderer_pretrained_dict)

        encoder.load_state_dict(encoder.state_dict())
        renderer.load_state_dict(renderer.state_dict())
        noise_renderer.load_state_dict(noise_renderer.state_dict())

    if render_optimizer:
        render_optimizer.load_state_dict(checkpoint['render_optim_dict'])

    if encoder_optimizer:
        encoder_optimizer.load_state_dict(checkpoint['encoder_optim_dict'])

    return checkpoint
