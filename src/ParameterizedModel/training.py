import time
import torch
import numpy as np
import warnings
from torch.utils.data import Dataset
from typing import Callable, Any


class Trainer():

    def __init__(self,
            model:Callable, optimizer:Callable, dataloader:Callable,
            accumulate:bool = False, scheduler:Callable = None, device:str = 'cpu'):

        self.optimizer = optimizer
        self.dataloader = dataloader
        self.model = model
        self.device = device
        self.scheduler = scheduler
        self.epochs = 0
        self.accumulate = accumulate
        self.total_loss = 0      
        self.loss_list = []  

    def training_step(self, batch):
        index, data = batch
        my_loss = self.model.loss(index, data)
        
        self.total_loss += my_loss.clone().detach()
        
        return my_loss

    def train(self, epochs = 1, feedback_every_epoch = 1, timeout = None):

        self.start = time.time()
        self.timeout = timeout

        for epoch in range(epochs):
            epoch = epoch + 1
            self.epochs = epoch
            loss = self.train_epoch()
            self.loss_list.append(loss)
            if np.isnan(loss):
                break

            addendum = ''

            if np.diff(self.loss_list[-2:]) == 0:
                # We did not make any progress during training
                if not self.scheduler is None:
                    for _ in range(len(self.dataloader)):
                        self.scheduler.step()
                    addendum += f' (made no progress, decreasing learning rate)'

            if epoch == 1 or epoch % feedback_every_epoch == 0:
                try:
                    print(f'{epoch}/{epochs} --- Mean Loss : {loss}; {self.model.info} + addendum')
                except:
                    print(f'{epoch}/{epochs} --- Mean Loss : {loss}' + addendum)

            if not self.timeout is None:
                if time.time() - self.start > self.timeout:
                    print('Timeout')
                    break

            

    def train_epoch(self):

        total_loss = 0
        accumulated_loss = None
        N = len(self.dataloader)
        for data in self.dataloader:
            loss = self.model.loss(data)
            
            if self.accumulate:
                if accumulated_loss is None:
                    accumulated_loss = loss
                else:
                    accumulated_loss += loss
            else:
                self.optimizer.zero_grad()
                loss.backward(retain_graph=False)
                self.optimizer.step()

            total_loss += loss.detach().clone() / N
            
            if torch.isnan(loss):
                warnings.warn('Loss is NaN')
                break

        if self.accumulate:
            self.optimizer.zero_grad()
            accumulated_loss.backward(retain_graph=False)
            self.optimizer.step()
        try:
            total_loss = total_loss.cpu().data.numpy().item()
        except:
            pass
        if not self.scheduler is None:
            self.scheduler.step()
        try:
            self.model.on_epoch_end()
        except:
            pass

        return total_loss