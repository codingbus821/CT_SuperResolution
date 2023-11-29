import os
import utility
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
import time
# import vessls
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from model import redcnn, drunet, dncnn
from loss import vggloss

class Trainer():
    def __init__(self, config, loader, ckp):
        self.config = config
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        
        ## Customize your loss function
        # self.loss = nn.L1Loss()
        self.loss = vggloss.VGGPerceptualLoss().to('cuda')
                    
        # self.model = redcnn.REDCNN().to('cuda')
        # self.model = dncnn.FDnCNN().to('cuda')

        self.model = redcnn.REDCNN().to('cuda')
        # self.model = drunet.UNetRes().to('cuda')
        self.optimizer = utility.make_optimizer(config['optimizer'], self.model)
        
        self.scheduler = MultiStepLR(self.optimizer, 
                                    milestones=config['optimizer']['milestones'], 
                                    gamma=config['optimizer']['gamma'])
        print('total number of parameter is {}'.format(sum(p.numel() for p in self.model.parameters())))

    def train(self):
        epoch = self.scheduler.last_epoch
        # self.ckp.add_train_log(torch.zeros(1))
        learning_rate = self.scheduler.get_last_lr()[0]
        self.model.train()
        train_loss = utility.Averager()
        timer = utility.Timer()
        for batch, (ldct, ndct) in enumerate(self.loader_train):
            ldct, ndct = self.prepare(ldct, ndct)
            
            ldct = utility.normalize(ldct, -500, 500)
            ndct = utility.normalize(ndct, -500, 500)

            denoised = self.model(ldct)
            loss = self.loss(denoised, ndct)
            
            self.model.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss.add(loss.item())

            self.optimizer.step()

            print(f"train loss : {train_loss.item()} / train time : {timer.t()} / learning rate : {learning_rate}")
        # vessl.log(step=epoch, payload={'train_loss': train_loss.item(), 'train_time': timer.t(), 'learning_rate': learning_rate})
        self.scheduler.step()

    def eval(self):
        epoch = self.scheduler.last_epoch
        if epoch % self.config['test_every'] == 0:
            self.ckp.add_val_log(torch.zeros(1))
            self.model.eval()
            timer = utility.Timer()
            
            with torch.no_grad():
                for i, (ldct, ndct) in enumerate(self.loader_test):
                    ldct, ndct = self.prepare(ldct, ndct)

                    ldct = utility.normalize(ldct, -500, 500)

                    denoised = self.model(ldct)
                    denoised = utility.denormalize(denoised, -500, 500)
                    
                    self.ckp.val_log[-1] += utility.calc_rmse(denoised, ndct) / len(self.loader_test)
    
                best = self.ckp.val_log.min(0) # best[0] is the minimum value, best[1] is the index of the minimum value
                print(f"eval val_log : {self.ckp.val_log[-1]} / val_time : {timer.t()}")
                # vessl.log(step=epoch//self.config['test_every']-1, payload={'val_rmse': self.ckp.val_log[-1], 'val_time': timer.t()})
                self.ckp.save(self.model, is_best=(best[1] + 1 == epoch // self.config['test_every']))

    def prepare(self, *args):
        device = torch.device('cpu' if self.config['cpu'] else 'cuda')
        def _prepare(tensor):
            return tensor.to(device)
        return [_prepare(a) for a in args]

    def terminate(self):
        epoch = self.scheduler.last_epoch
        return epoch >= self.config['epochs']
