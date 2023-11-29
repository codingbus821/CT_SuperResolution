from importlib import import_module
from torch.utils.data import DataLoader
import os

class Data:
    def __init__(self, config, test_only=False):
        self.loader_train = None
        if not test_only:
            # main에서 data.Data(config)로 불러와짐
            self.loader_train = None
            module_train = import_module('data.srdata')
            trainset = getattr(module_train, 'SRData')(config, mode='train', 
                                                        augment=config['dataset']['augment'])
            self.loader_train = DataLoader(
                trainset,
                batch_size=config['dataset']['batch_size'],
                num_workers=config['n_threads'],
                shuffle=True,
                pin_memory=True
            )
            module_test = import_module('data.srdata')
            testset = getattr(module_test, 'SRData')(config, mode='valid')
            self.loader_test = DataLoader(
                testset,
                batch_size=1,
                num_workers=config['n_threads'],
                shuffle=False,
                pin_memory=True
            )

        else:
            module_test = import_module('data.srdata')
            testset = getattr(module_test, 'SRData')(config, mode='test')
            self.loader_test = DataLoader(
                testset,
                batch_size=1,
                num_workers=config['n_threads'],
                shuffle=False,
                pin_memory=True
            )

