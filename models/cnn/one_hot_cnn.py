""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.search_cells import SearchCell
from models.cnn.architect import Architect
from models.cnn import ops
from torch.nn.parallel._functions import Broadcast
from models.cnn.search_cnn import SearchCNNController, SearchCNN
from visualize import plot
import genotypes as gt
import logging
import numpy as np
import json


class OneHotSearchCNNController(SearchCNNController):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, **kwargs):        
        SearchCNNController.__init__(self, **kwargs)
        self.aux = float(kwargs['one-hot']['aux weight'])
        if self.aux > 0.0:
           raise NotImplementedError('Aux')
        if kwargs['one-hot']['genotype path'] == 'random-simple':
            self.weights_reduce = []
            self.weights_normal = []
            for alpha in self.alpha_reduce:
               self.weights_reduce.append(np.random.randint(low=0, high=alpha.shape[1], size=alpha.shape[0]).tolist())
            for alpha in self.alpha_normal:
               self.weights_normal.append(np.random.randint(low=0, high=alpha.shape[1], size=alpha.shape[0]).tolist())

        else:
            with open(kwargs['one-hot']['genotype path'].format(kwargs['seed'])) as inp:
                self.weights_reduce, self.weights_normal = json.loads(inp.read())
            
    def train_step(self, trn_X, trn_y, val_X, val_y):
        lr = self.lr_scheduler.get_last_lr()[0]
        # phase 1. child network step (w)
        self.w_optim.zero_grad()
        loss = self.loss(trn_X, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(self.weights(), self.w_grad_clip)
        self.w_optim.step()        
        return loss

    def forward(self, x):        
        return self.net(x, self.weights_normal, self.weights_reduce)
    
