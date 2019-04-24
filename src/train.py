import os
import re
import cv2
import json
import numpy as np

import argus
from argus import Model, load_model
from argus.engine import State
from argus.callbacks import MonitorCheckpoint, EarlyStopping, LoggingToFile

from src.utils import rle_decode

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

from src.models.unet_flex import UNetFlexProb
from src.losses import ShipLoss
from src.metrics import ShipIOUT
from src.utils import  filename_without_ext, get_best_model_path
from src.transforms import ProbOutputTransform, test_transforms, train_transforms
from src.dataset import ShipDatasetFolds
from src.lr_scheduler import ReduceLROnPlateau
from src.models.resnet_blocks import resnet34


os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

EPOCHS = 150
BATCH_SIZE = 32
LR = 1e-4

SAVE_DIR = '/workdir/data/models/linknet34_folds_003'
folds_path = '/workdir/data/datasets/kfolds_small.pk'
imgs_dir = '/workdir/data/datasets/ships_small/train_small/images/'
trgs_dir = '/workdir/data/datasets/ships_small/train_small/targets/'

IMG_EXT = '.jpg'
TRG_EXT = '.png'

FOLDS = list(range(3))
print(FOLDS)

IMG_SIZE = (256, 256)
SKIP_EMPTY_PROB = 0.9

train_trns = train_transforms(size=IMG_SIZE, skip_empty_prob=SKIP_EMPTY_PROB, sigma_g=10)
val_trns = test_transforms(size=IMG_SIZE)

def get_data_loaders(batch_size, train_folds, val_folds):
    train_dataset = ShipDatasetFolds(folds_path, train_folds, imgs_dir=imgs_dir, trgs_dir=trgs_dir, masks=True, **train_trns)
    val_dataset = ShipDatasetFolds(folds_path, val_folds, imgs_dir=imgs_dir, trgs_dir=trgs_dir, masks=True, **val_trns)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8)
    return train_loader, val_loader


class ShipMetaModel(Model):
    nn_module = {
        'UNetFlexProb': UNetFlexProb,
    }
    loss = {
        'ShipLoss': ShipLoss
    }
    prediction_transform = {
        'ProbOutputTransform': ProbOutputTransform
    }

params = {'nn_module': ('UNetFlexProb', {
            'num_classes': 5,
            'num_channels': 3,
            'blocks': resnet34,
            'final': 'sigmoid',
            'skip_dropout': True,
            'dropout_2d': 0.2,
            'is_deconv': True,
            'pretrain': 'resnet34',
            'pretrain_layers': [True for _ in range(5)]
            }),
        'loss': ('ShipLoss', {
            'fb_weight': 0.25,  # Need tuning!
            'fb_beta': 1,
            'bce_weight': 0.25,
            'prob_weight': 0.25,
            'mse_weight': 0.25
            }),
        'prediction_transform': ('ProbOutputTransform', {
            'segm_thresh': 0.5,
            'prob_thresh': 0.5
            }),
        'optimizer': ('Adam', {'lr': LR}),
        'device': 'cuda'
    }

def train_fold(save_path, train_folds, val_folds):
    train_loader, val_loader = get_data_loaders(BATCH_SIZE,
                                                train_folds, val_folds)
    model = ShipMetaModel(params)
    callbacks = [MonitorCheckpoint(save_path, monitor='val_iout', max_saves=2, copy_last=True),
             EarlyStopping(monitor='val_iout', patience=40),
             ReduceLROnPlateau(monitor='val_iout', patience=10, factor=0.2, min_lr=1e-8),
             LoggingToFile(os.path.join(save_path, 'log.txt'))]

    model.fit(train_loader,
          val_loader=val_loader,
          max_epochs=EPOCHS,
          callbacks=callbacks,
          metrics=['iout'])

if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    else:
        print(f"Folder {SAVE_DIR} already exists.")

    with open(os.path.join(SAVE_DIR, 'source.py'), 'w') as outfile:
        outfile.write(open(__file__).read())

    with open(os.path.join(SAVE_DIR, 'params.json'), 'w') as outfile:
        json.dump(params, outfile)

    for i in range(len(FOLDS)):
        val_folds = [FOLDS[i]]
        train_folds = FOLDS[:i] + FOLDS[i + 1:]
        save_fold_dir = os.path.join(SAVE_DIR, f'fold_{FOLDS[i]}')
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds)
