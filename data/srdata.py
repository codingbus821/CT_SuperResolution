import os
import glob
from data import common
import numpy as np
import torch.utils.data as data
import torch
import math
import random
import pydicom
import mat73
from pathlib import Path
import time

from matplotlib import pyplot as plt

class SRData(data.Dataset):
    def __init__(self, config, mode='train', augment=False):
        self.dataset_spec = config['dataset']
        self.mode = mode
        self.augment = augment
        self.device = torch.device('cpu' if config['cpu'] else 'cuda')
        self.ldct, self.ndct = self._scan()

    def __getitem__(self, idx):
        ldct, ndct = self._load_file(idx)
        ldct, ndct = self.preparation(ldct, ndct)
        return ldct.copy(), ndct.copy()

    def __len__(self):
        return self.ldct.shape[2]

    def _scan(self):
        ldct = np.array([]).reshape(512,512,0)
        ndct = np.array([]).reshape(512,512,0)
        patient_files = glob.glob(os.path.join(self.dataset_spec['data_dir'], self.mode, '*.mat'))
        for i in patient_files:
            ldct = np.concatenate((ldct, mat73.loadmat(i)['f_qd']), 2)
            ndct = np.concatenate((ndct, mat73.loadmat(i)['f_nd']), 2)
        u_water = 0.0192
        # mm-1 to HU
        ldct = (ldct - u_water) * 1000 / u_water
        ndct = (ndct - u_water) * 1000 / u_water
        
        return ldct, ndct

    def _load_file(self, idx):        
        ldct = self.ldct[:, :, idx].astype(np.float32)
        ndct = self.ndct[:, :, idx].astype(np.float32)
        return ldct, ndct

    def preparation(self, ldct, ndct):
        ldct, ndct = np.expand_dims(ldct, 0), np.expand_dims(ndct, 0)
        if self.mode == 'train':
            if self.augment:
                ldct, ndct = common.get_patch(ldct, ndct, patch_size=64)
                ldct, ndct = common.augment(ldct, ndct)
        return ldct, ndct
