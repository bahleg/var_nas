""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn_var_local.search_cells import SearchCell
from models.cnn_var_local.ops import LocalVarConv2d, LocalVarLinear
from torch.nn.parallel._functions import Broadcast

from visualize import plot
import logging

import numpy as np


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class LVarSearchCNN(nn.Module):
    """ Search CNN model """

    def __init__(self,  primitives,  C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3, t=1.0, stochastic_w=True, stochastic_gamma=True):
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

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            LocalVarConv2d(C_in, C_cur, 3, 1, 1, bias=False),
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

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur,
                              reduction_p, reduction, primitives)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = LocalVarLinear(C_p, n_classes)

        self.t = float(t)
        self.init_t = float(t)
        self.q_gamma_normal = nn.ParameterList()
        self.q_gamma_reduce = nn.ParameterList()
        n_ops = len(primitives)

        for i in range(n_nodes):
            self.q_gamma_normal.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.q_gamma_reduce.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
        self.stochastic_w = stochastic_w
        self.stochastic_gamma = stochastic_gamma

        if stochastic_w:
            self.disable_w()

    def forward(self, x):
        s0 = s1 = self.stem(x)
        t = torch.ones(1).to(x.device)*self.t

        for cell in self.cells:
            gammas = self.q_gamma_reduce if cell.reduction else self.q_gamma_normal
            if self.stochastic_gamma:
                weights = [torch.distributions.RelaxedOneHotCategorical(
                    t, logits=gamma).rsample([x.shape[0]]) for gamma in gammas]
            else:
                weights = [gamma for gamma in gammas]

            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits

    def disable_w(self):
        all_ = [self]
        i = 0
        while i < len(all_):
            current = all_[i]
            i += 1
            try:
                for c in current.children():
                    all_ += [c]
            except:
                pass
        for c in all_:
            if 'stochastic' in c.__dict__:
                c.stochastic = False


# https://github.com/pytorch/pytorch/blob/master/torch/distributions/kl.py#L405
def kl_normal_normal(pl, ql, ps, qs):

    #ps = ps.view(-1)
    #qs = qs.view(-1)
    #pl = pl.view(-1)
    #ql = ql.view(-1)

    var_ratio = (ps / qs).pow(2)

    t1 = ((pl - ql) / qs).pow(2)

    result = 0.5 * (var_ratio + t1 - 1 - var_ratio.log()).sum()
    return result


class LVarSearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self,  **kwargs):
        super().__init__()
        subcfg = kwargs['var_darts']
        C_in = int(subcfg['input_channels'])
        C = int(subcfg['init_channels'])
        n_classes = int(subcfg['n_classes'])
        n_layers = int(subcfg['layers'])
        n_nodes = int(subcfg['n_nodes'])
        stem_multiplier = int(subcfg['stem_multiplier'])
        self.device = kwargs['device']
        self.dataset_size = int(subcfg['dataset size'])
        self.n_nodes = n_nodes
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.stochastic_w = int(subcfg['stochastic w']) != 0
        self.stochastic_gamma = int(subcfg['stochastic gamma']) != 0

        # initialize architect parameters: alphas
   
        self.delta = float(subcfg['delta'])

        self.sample_num = int(subcfg['sample num'])
        if subcfg['primitives'] == 'DARTS':
            primitives = [
                'max_pool_3x3',
                'avg_pool_3x3',
                'skip_connect', # identity
                'sep_conv_3x3',
                'sep_conv_5x5',
                'dil_conv_3x3',
                'dil_conv_5x5',
                'none'
            ] 
        elif subcfg['primitives'] == 'DARTS non-zero':
            primitives = [
                'max_pool_3x3',
                'avg_pool_3x3',
                'skip_connect', # identity
                'sep_conv_3x3',
                'sep_conv_5x5',
                'dil_conv_3x3',
                'dil_conv_5x5',
            ]            
        else:
            raise ValueError('Incorrect value for primitives')
        n_ops = len(primitives)
        self.net = LVarSearchCNN(primitives, C_in, C,  n_classes,
                                 n_layers, n_nodes, stem_multiplier, t=float(subcfg['initial temp']), stochastic_gamma=self.stochastic_gamma,
                                 stochastic_w=self.stochastic_w)

        self.alpha_h = nn.ParameterList()
        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()
        self.alpha_w_h = {}
        self.sigmas_w = {}

        for w in self.net.parameters():
            if 'sigma' in w.__dict__:  # var!
                self.alpha_h.append(nn.Parameter(torch.zeros(w.shape)))
                self.alpha_w_h[w] = self.alpha_h[-1]
                self.sigmas_w[w] = w.sigma

        for i in range(n_nodes):
            self.alpha_normal.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        if not self.stochastic_w:
            self.net.disable_w()

        self.all_params = list(self.net.parameters()) + list(self.alpha_h)

        # weights optimizer
        self.w_optim = torch.optim.SGD(self.all_params, float(subcfg['optim']['w_lr']), momentum=float(subcfg['optim']['w_momentum']),
                                       weight_decay=float(subcfg['optim']['w_weight_decay']))

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.w_optim, int(kwargs['epochs']), eta_min=float(subcfg['optim']['w_lr_min']))
        self.w_grad_clip = float(subcfg['optim']['w_grad_clip'])

    def writer_callback(self, writer, epoch, cur_step):
        pass

    def train_step(self, trn_X, trn_y, val_X, val_y):
        lr = self.lr_scheduler.get_last_lr()[0]
        # phase 1. child network step (w)
        self.w_optim.zero_grad()
        loss = self.loss(trn_X, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(self.all_params, self.w_grad_clip)
        self.w_optim.step()
        return loss

    def new_epoch(self, e, writer, l):
        self.lr_scheduler.step(epoch=e)   
        self.net.t = self.net.init_t + self.delta*e

    def forward(self, x):
        return self.net(x)

    def loss(self, X, y):
        logits = self.forward(X)
        kld = self.kld()
        return kld / self.dataset_size + self.criterion(logits, y)

    def kld(self):
        k = 0

        if self.stochastic_w:
            for w, h in self.alpha_w_h.items():
                k += kl_normal_normal(w, torch.zeros_like(w),
                                      0.01+torch.exp(self.sigmas_w[w]), 0.01+torch.exp(h))

        if self.stochastic_gamma:
            t = torch.ones(1).to(self.device)*self.net.t
            for a, ga in zip(self.alpha_normal, self.net.q_gamma_normal):
                g = torch.distributions.RelaxedOneHotCategorical(
                    t, logits=ga)
                sample = (g.rsample([self.sample_num])+0.0001)
                k += (g.log_prob(sample)).sum() / self.sample_num

            for a, ga in zip(self.alpha_reduce, self.net.q_gamma_reduce):
                g = torch.distributions.RelaxedOneHotCategorical(
                    t, logits=ga)
                sample = (g.rsample([self.sample_num])+0.0001)
                k += (g.log_prob(sample)).sum() / self.sample_num

        return k

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        if True:
            logger.info("####### GAMMA #######")
            logger.info("# Gamma - normal")

            for alpha in self.net.q_gamma_normal:
                logger.info(alpha)

            logger.info("\n# Gamma - reduce")
            for alpha in self.net.q_gamma_reduce:
                logger.info(alpha)
            logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        raise NotImplementedError()
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2+self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

  
    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def plot_genotype(self, plot_path, caption):
        plot(self.genotype().normal, plot_path+'-normal', caption+'-normal')
        plot(self.genotype().reduce, plot_path+'-reduce', caption+'-reduce')
