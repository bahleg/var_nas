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
        #self.log_t = None                 
        SearchCNNController.__init__(self, **kwargs)
        #self.log_t =  torch.nn.Parameter(torch.zeros(1))
        self.e_alpha = (torch.ones(1)*(float(kwargs['darts entropy']['expected entropy']))).to(self.device)        
        self.e_lam = float(kwargs['darts entropy']['entropy regularizer coef'])
        self.e_sample_num  = int(kwargs['darts entropy']['entropy sample num'])
        subcfg = kwargs['darts']
        self.alpha_optim = torch.optim.Adam(self.alphas(), float(subcfg['optim']['alpha_lr']), betas=(0.5, 0.999),
                                    weight_decay=float(subcfg['optim']['alpha_weight_decay']))
    def alphas(self):
        for n, p in self._alphas:
            yield p
        #if self.log_t is not None:
        #    yield self.log_t

    def calc_entropy_cat(self):
        entropy = 0
        for alpha in list(self.alpha_reduce) + list(self.alpha_normal):
            for subalpha in alpha:
                    distr = torch.distributions.categorical.Categorical(logits=subalpha)
                    entropy += distr.entropy()
        return entropy
    def calc_entropy_cat_igr(self):
        entropy = 0
        SSIZE = 30
        for alpha, cov in zip(self.alpha_reduce, self.alpha_cov_reduce):
            for subalpha, subcov in zip(alpha, cov):
                distr = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(subalpha,     
                                                                                              subcov,
                                                                                              torch.ones(subalpha.shape[0]).to(self.device))
                entropy +=  torch.distributions.categorical.Categorical(probs=torch.nn.functional.softmax(distr.rsample([SSIZE])/0.01).mean(0)).entropy()
        return entropy
        
                        
    def calc_entropy(self):
        if self.sampling_mode == 'igr':
              return self.calc_entropy_cat_igr()
        return self.calc_entropy_cat()

        if self.sampling_mode == 'gumbel-softmax':
            return self.calc_entropy_gs()
        elif self.sampling_mode == 'igr':
            return self.calc_entropy_igr()
        else:
            raise ValueError('Bad sampling mode')
            

    def calc_entropy_gs(self):
        entropy = 0
        for alpha in list(self.alpha_reduce) + list(self.alpha_normal):
            for subalpha in alpha:                
                    distr = torch.distributions.RelaxedOneHotCategorical(self.t, logits=subalpha)
                    subentropy = 0
                    for _ in range(self.e_sample_num):
                        subentropy -= distr.log_prob(distr.rsample())
                    subentropy /= self.e_sample_num
                    entropy += subentropy
                
        return entropy

    #https://arxiv.org/pdf/1912.09588.pdf
    
    def log_det_jac(self, logits, t):
        k = len(logits)    
        exp_logit = torch.exp(logits/t)
        eps = 0.01
        part1 = -2*(k-1) * torch.log((exp_logit.sum()))
        
        part2 = torch.log(abs(1-t*(exp_logit/(eps+logits)).sum()))
        
        part3 = -(k-1) * torch.log(t)
        
        part4 = (torch.log(abs(logits))+abs(logits/t)).sum() 
        
        return part1 + part2 + part3 +part4

    def calc_entropy_igr(self):
        entropy = 0
        for alpha, cov in zip(self.alpha_reduce, self.alpha_cov_reduce):
            for subalpha, subcov in zip(alpha, cov):
                distr = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(subalpha,     
                                                                                              subcov,
                                                                                              torch.ones(subalpha.shape[0]).to(self.device))
                subentropy = 0
                for _ in range(self.e_sample_num):
                    sample = distr.sample()                
                    subentropy=subentropy -distr.log_prob(sample) +self.log_det_jac(sample, self.t)
                subentropy /= self.e_sample_num
                entropy += subentropy
        return entropy

    def forward(self, x):
        if self.sampling_mode == 'gumbel-softmax':
            weights_normal = [torch.distributions.RelaxedOneHotCategorical(
                    self.t, logits=alpha).rsample([x.shape[0]]) for alpha in self.alpha_normal]
            weights_reduce = [torch.distributions.RelaxedOneHotCategorical(
                    self.t, logits=alpha).rsample([x.shape[0]]) for alpha in self.alpha_reduce]   
        elif self.sampling_mode == 'igr':

            weights_normal = []
            weights_reduce = []

            for alpha, cov in zip(self.alpha_normal, self.alpha_cov_normal):  
                subsample = []             
                for subalpha, subcov in zip(alpha, cov):                    
                    subsample.append([])
                    distr = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(subalpha,     
                                                                                              subcov,
                                                                                              torch.ones(subalpha.shape[0]).to(self.device))
                    sample = distr.rsample([x.shape[0]])
                    subsample[-1].append(F.softmax(sample/self.t, dim=-1))                              
                weights_normal.append(torch.stack([torch.cat(s, 1) for s in subsample], 1))                   

            for alpha, cov in zip(self.alpha_reduce, self.alpha_cov_reduce):  
                subsample = []             

                for subalpha, subcov in zip(alpha, cov):                    
                    subsample.append([])
                    distr = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(subalpha,     
                                                                                              subcov,
                                                                                              torch.ones(subalpha.shape[0]).to(self.device))
                    sample = distr.rsample([x.shape[0]])
                    subsample[-1].append(F.softmax(sample/self.t, dim=-1))                              
                weights_reduce.append(torch.stack([torch.cat(s, 1) for s in subsample], 1))                    
                                
        else:
            raise ValueError('Bad sampling mode')
                             
        return self.net(x, weights_normal, weights_reduce)



    def loss(self, X, y):
        logits = self.forward(X)                     
        return  self.criterion(logits, y) + self.e_lam*(self.e_alpha  - self.calc_entropy()) ** 2 
        #return self.e_lam*(self.e_alpha  - self.calc_entropy()) ** 2


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
        
        logger.info(self.t)
        logger.info("#####################")

        logger.info("\n# Entropy")
        
        logger.info( self.calc_entropy())
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def new_epoch(self, e, w, l):
        self.lr_scheduler.step(epoch=e)    
        self.t = self.init_t + self.delta_t*e
        self.t = torch.tensor(self.t).to(self.device)
        self.print_alphas(l)
