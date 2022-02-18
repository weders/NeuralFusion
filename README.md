# NeuralFusion

This is the official implementation of [NeuralFusion: Online Depth Map Fusion in Latent Space](silvanweder.com/publications/neural-fusion/). We provide code to train the proposed pipeline on ShapeNet, ModelNet, as well as Tanks and Temples.

If you plan to use NeuralFusion for commercial purposes, please contact the author first. For more information, please also see the license.

If you find our code or paper useful, please consider citing

```
@InProceedings{Weder_2021_CVPR,
    author    = {Weder, Silvan and Schonberger, Johannes L. and Pollefeys, Marc and Oswald, Martin R.},
    title     = {NeuralFusion: Online Depth Fusion in Latent Space},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3162-3172}
}
```

## Installation

Install the code using the following steps

```
conda env create -f environment.yml
conda activate neural-fusion
```

## Data Preparation

In order to prepare the data, please follow the instructions explained in [this](https://github.com/weders/RoutedFusion) repo.


## Training

In order to train the pipeline, run the following

```
python train.py --experiment_path /path/where/you/want/to/save/the/experiment \ 
                --data_path /path/to/your/data \
                --config configs/train/your/config.yaml
```

## Testing

In order to test the pipeline, run the following

```
python test.py --test /path/to/your/test/config \
               --root_path /path/where/you/saved/your/experiments \
               --data_path /path/to/your/data \
               --experiment $experiment_name \ 
               --version $experiment_version \
               --checkpoint $experiment_checkpoint
```

For example, if you would like to test the pretrained on ShapeNet, you need to run the following command

```
export DATA_PATH=/path/to/your/preprocessed/shapenet/data


python test.py --test configs/test/shapenet/shapenet.noise.005.yaml \
               --root_path pretrained_models \
               --data_path $DATA_PATH \
               --experiment shapenet_noise_005 \ 
               --version 0 \
               --checkpoint best.ckpt
```
