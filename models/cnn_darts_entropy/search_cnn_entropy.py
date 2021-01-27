from ..cnn.search_cnn import SearchCNNController
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.search_cells import SearchCell
from models.cnn.architect import Architect
from models.cnn import ops 
from torch.nn.parallel._functions import Broadcast
from visualize import plot
import genotypes as gt
import logging
import numpy as np

MIN_ALPHA = 0.2 

class SearchCNNControllerWithEntropy(SearchCNNController):

    def __init__(self, **kwargs):
        self.log_t = None                 
        SearchCNNController.__init__(self, **kwargs)
        self.log_t =  torch.nn.Parameter(torch.zeros(1))
        self.e_alpha = (torch.ones(1)*(float(kwargs['darts entropy']['expected entropy']))).to(self.device)        
        self.e_lam = float(kwargs['darts entropy']['entropy regularizer coef'])
        self.e_sample_num  = int(kwargs['darts entropy']['entropy sample num'])
        subcfg = kwargs['darts']
        self.alpha_optim = torch.optim.Adam(self.alphas(), float(subcfg['optim']['alpha_lr']), betas=(0.5, 0.999),
                                    weight_decay=float(subcfg['optim']['alpha_weight_decay']))
    def alphas(self):
        for n, p in self._alphas:
            yield p
        if self.log_t is not None:
            yield self.log_t
    

    def calc_entropy(self):
        entropy = 0
        for alpha in list(self.alpha_reduce) + list(self.alpha_normal):
            for subalpha in alpha:                
                if self.sampling_mode == 'gumbel-softmax':
                    distr = torch.distributions.RelaxedOneHotCategorical(MIN_ALPHA+torch.exp(self.log_t), logits=subalpha)
                subentropy = 0
                for _ in range(self.e_sample_num):
                    subentropy -= distr.log_prob(distr.rsample())
                subentropy /= self.e_sample_num
                entropy += subentropy
                
        return entropy

    def forward(self, x):
        if self.sampling_mode == 'gumbel-softmax':
            weights_normal = [torch.distributions.RelaxedOneHotCategorical(
                    MIN_ALPHA+torch.exp(self.log_t), logits=alpha).rsample([x.shape[0]]) for alpha in self.alpha_normal]
            weights_reduce = [torch.distributions.RelaxedOneHotCategorical(
                    MIN_ALPHA+torch.exp(self.log_t), logits=alpha).rsample([x.shape[0]]) for alpha in self.alpha_reduce]                    
        return self.net(x, weights_normal, weights_reduce)


    def loss(self, X, y):
        logits = self.forward(X)             
        return   self.criterion(logits, y) + self.e_lam*(self.e_alpha  - self.calc_entropy()) ** 2 


    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        logger.info("\n# t")
        
        logger.info( MIN_ALPHA+torch.exp(self.log_t))
        logger.info("#####################")

        logger.info("\n# Entropy")
        
        logger.info( self.calc_entropy())
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def new_epoch(self, e, w, l):
        self.lr_scheduler.step(epoch=e)    
        self.t = self.t + self.delta_t*e
        self.print_alphas(l)