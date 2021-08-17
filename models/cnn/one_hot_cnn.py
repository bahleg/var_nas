""" CNN for model training """
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


class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size-5, padding=0, count_include_pad=False), # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False), # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


class OneHotCNN(nn.Module):
    """ Search CNN model """

    def __init__(self, primitives,  C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3, drop=0.0, aux=0.0, input_size=8):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.aux = aux 
        if aux > 0.0:
            self.aux_pos = 2*n_layers//3 
            
        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes,   C_pp, C_p, C_cur,
                              reduction_p, reduction, primitives, drop)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out
            if aux>0 and i == self.aux_pos:
                self.aux_head = AuxiliaryHead(input_size//4, C_p, n_classes)
                
                

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)
        aux_logits = None
        
        for i, cell in enumerate(self.cells):
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
            if self.aux and  i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        if self.training:
            return logits, aux_logits
        else:
            return logits
        
        

class OneHotSearchCNNController(SearchCNNController):
    """ SearchCNN controller supporting multi-gpu """

    def init_net(self, kwargs):
        subcfg = kwargs['darts']
        #drop = float(subcfg['drop path proba'])
        C_in = int(subcfg['input_channels'])
        C = int(subcfg['init_channels'])
        n_classes = int(subcfg['n_classes'])
        n_layers = int(subcfg['layers'])
        n_nodes = int(subcfg['n_nodes'])
        stem_multiplier = int(subcfg['stem_multiplier'])
        
        primitives = self.get_primitives(kwargs)

        self.aux = float(kwargs['one-hot']['aux weight'])
        input_size = int(kwargs['one-hot']['input dim'])
        
        if self.aux > 0.0:
            self.aux_pos = 2*n_layers//3
        
        self.drop_init = float(subcfg['drop path proba initial'])
        self.drop_delta = float(subcfg['drop path proba delta'])
        
        if self.drop_init > 0 or self.drop_delta != 0:
            self.drop = torch.tensor(0.0001) # need to initialize cells properly
        else:
            self.drop = 0.0

        self.net = OneHotCNN(primitives, C_in, C, n_classes, n_layers,
                             n_nodes, stem_multiplier, self.drop, self.aux, input_size)
    
    def __init__(self, **kwargs):        
        SearchCNNController.__init__(self, **kwargs)
        
            
            
            
        if kwargs['one-hot']['genotype path'] == 'random-simple':
            self.weights_reduce = []
            self.weights_normal = []
            for alpha in self.alpha_reduce:
               self.weights_reduce.append(np.random.randint(low=0, high=alpha.shape[1], size=alpha.shape[0]).tolist())
            for alpha in self.alpha_normal:
               self.weights_normal.append(np.random.randint(low=0, high=alpha.shape[1], size=alpha.shape[0]).tolist())
            print (self.weights_reduce, self.weights_normal)
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
        
    def loss(self, X, y):
        logits, aux_logits = self.forward(X)
        if self.aux>0.0:
            return self.aux *  self.criterion(aux_logits, y) +  self.criterion(logits, y)
        return self.criterion(logits, y)
    

    def new_epoch(self, e, w, l):
        SearchCNNController.new_epoch(self, e, w, l)
        if self.drop_delta != 0.0 or self.drop_init > 0.0:
             self.drop.data *= 0
             self.drop.data += self.drop_init + self.drop_delta * e


