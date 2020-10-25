""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn_var_local.search_cells import SearchCell
from models.cnn_var_local.ops import LocalVarConv2d, LocalVarLinear
import genotypes as gt
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

    def __init__(self,  C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3, t=1.0, delta=-0.016):
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
                              reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = LocalVarLinear(C_p, n_classes)
        
        self.t = float(t)
        self.delta = float(delta)

        self.q_gamma_normal = nn.ParameterList()
        self.q_gamma_reduce = nn.ParameterList()
        n_ops = len(gt.PRIMITIVES)

        for i in range(n_nodes):
            self.q_gamma_normal.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.q_gamma_reduce.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
        
        self.stochastic_gamma = True
        self.stochastic_w = True
        self.pruned = False

    def disable_stochastic_w(self):
        logging.debug('disabling stochastic w')        
        self.stochastic_w = False
        all_ = [self]
        i = 0 
        while i<len(all_):
                current = all_[i]
                i+=1
                try:
                    for c in current.children():
                        all_+=[c]
                except:
                    pass
        for c in all_:
                if 'stochastic' in c.__dict__:
                    c.stochastic = False 

    def forward(self, x):
        s0 = s1 = self.stem(x)
        t = torch.ones(1).to(x.device)*self.t            
            
        for cell in self.cells:            
            gammas = self.q_gamma_reduce if cell.reduction else self.q_gamma_normal
            if  self.stochastic_gamma:
                weights = [torch.distributions.RelaxedOneHotCategorical(
                    t, logits=gamma).rsample([x.shape[0]]) for gamma in gammas]
            else:
                if self.t!=0:
                    weights = [torch.nn.functional.softmax(gamma/t) for gamma in gammas]
                else:
                    weights = gammas

            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits

    def prune(self, w, k=None):
        self.stochastic_gamma = False
        self.t = 0
        for edges in self.q_gamma_normal:
            edge_max, primitive_indices = torch.topk(                
                edges[:, :], 1) 
            edges.data *= 0
            if k:
                k_ = k
            else:
                k_ = edge_max.shape[0]

            topk_edge_values, topk_edge_indices = torch.topk(
                edge_max.view(-1), k_)
            node_gene = []

            for edge_idx in topk_edge_indices:
                edges.data[edge_idx, primitive_indices[edge_idx]] += 1
        for edges in self.q_gamma_reduce:
            edge_max, primitive_indices = torch.topk(
                edges[:, :], 1)  
            edges.data *= 0
            if k:
                k_ = k
            else:
                k_ = edge_max.shape[0]
            topk_edge_values, topk_edge_indices = torch.topk(
                edge_max.view(-1), k_)
            node_gene = []
            for edge_idx in topk_edge_indices:
                edges.data[edge_idx, primitive_indices[edge_idx]] += 1
        if w:
            all_ = [self]
            i = 0 
            while i<len(all_):
                current = all_[i]
                i+=1
                try:
                    for c in current.children():
                        all_+=[c]
                except:
                    pass
            for c in all_:
                if 'stochastic' in c.__dict__:
                    c.stochastic = False                   

#https://github.com/pytorch/pytorch/blob/master/torch/distributions/kl.py#L405
def kl_normal_normal(pl, ql, ps, qs):
    
    #ps = ps.view(-1)
    #qs = qs.view(-1)
    #pl = pl.view(-1)
    #ql = ql.view(-1)
    
    var_ratio = (ps / qs).pow(2)
    
    t1 = ((pl - ql) / qs).pow(2)
    
    result =  0.5 * (var_ratio + t1 - 1 - var_ratio.log()).sum()
    return result


class LVarSearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, device, **kwargs):
        super().__init__()
        C_in = int(kwargs['input_channels'])
        C = int(kwargs['init_channels'])
        n_classes = int(kwargs['n_classes'])
        n_layers = int(kwargs['layers'])
        n_nodes = int(kwargs['n_nodes'])
        stem_multiplier = int(kwargs['stem_multiplier'])
        device_ids = kwargs.get('device_ids', None)
        self.dataset_size = int(kwargs['dataset size'])
        self.n_nodes = n_nodes        
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.device = device

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        #self.log_t_h_mean = nn.Parameter(torch.ones(
        #    1) * (float(kwargs['initial temp log'])))
        #self.log_t_h_log_sigma = nn.Parameter(torch.ones(
        #    1) * float(kwargs['initial temp log sigma']))

        self.delta = float(kwargs['delta'])

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        self.alpha_w_h = {}
        self.sigmas_w = {}
        self.alpha_h = nn.ParameterList()

        self.sample_num = int(kwargs['sample num'])

        self.net = LVarSearchCNN(C_in, C,  n_classes,
                                 n_layers, n_nodes, stem_multiplier, kwargs['initial temp'], kwargs['delta'])

        for w in self.net.parameters():
            if 'sigma' in w.__dict__:
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

        

        self.net.stochastic_gamma =  int(kwargs['stochastic_gamma'])!=0
        self.net.stochastic_w =  int(kwargs['stochastic_w'])!=0
        if not self.net.stochastic_w:
            self.net.disable_stochastic_w()        

    
    def writer_callback(self, writer, cur_step):
        hist_values = []
        for val in self.net.q_gamma_normal:
            hist_values.extend(F.softmax(val).cpu().detach().numpy().tolist())
        hist_values = np.array(hist_values).flatten().tolist()
        writer.add_histogram('train/gamma_normal', hist_values, cur_step)# %%
        
        hist_values = []
        for val in self.net.q_gamma_reduce:
            hist_values.extend(F.softmax(val).cpu().detach().numpy().tolist())
        hist_values = np.array(hist_values).flatten().tolist()
        writer.add_histogram('train/gamma_reduce', hist_values, cur_step)# %%        
        
        
        #hist_values = self.net.log_q_t_mean.cpu().detach().numpy() + np.random.randn(10000)*torch.exp(self.net.log_q_t_log_sigma).detach().cpu().numpy()
        #writer.add_histogram('train/temp', np.exp(hist_values), cur_step, bins='auto') # %%        
        #writer.add_histogram('train/log_temp', hist_values, cur_step, bins='auto') 
        
        #writer.add_scalar('train/temp_min', torch.exp(self.net.log_q_t_mean-2*torch.exp(self.net.log_q_t_log_sigma)).cpu().detach().numpy(), cur_step)
        #writer.add_scalar('train/temp_max', torch.exp(self.net.log_q_t_mean+2*torch.exp(self.net.log_q_t_log_sigma)).cpu().detach().numpy(), cur_step)
        


    def new_epoch(self, e, writer):
        self.net.t += self.net.delta
        

    def forward(self, x):
        return self.net(x)

    def loss(self, X, y):
        logits = self.forward(X)        
        kld = self.kld()    
        return kld / self.dataset_size + self.criterion(logits, y)
        
    def kld(self):
        k = 0
        if self.net.stochastic_w:
            for w, h in self.alpha_w_h.items():                        
                k+= kl_normal_normal( w, torch.zeros_like(w), torch.exp(self.sigmas_w[w]), torch.exp(h))    
        
        if self.net.stochastic_gamma:
            t = torch.ones(1).to(self.device)*self.net.t            
            for a, ga in zip(self.alpha_normal, self.net.q_gamma_normal):
                g = torch.distributions.RelaxedOneHotCategorical(
                    t, logits=ga)                                 
                sample = (g.rsample([self.sample_num])+0.0001)
                k += (g.log_prob(sample)).sum() / self.sample_num

            for a, ga in zip(self.alpha_normal, self.net.q_gamma_reduce):
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
    
        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        logger.info("####### GAMMA #######")
        logger.info("# Gamma - normal")

        for alpha in self.net.q_gamma_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Gamma - reduce")
        for alpha in self.net.q_gamma_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

            #logger.info('Temp: {}...{}'.format(str(torch.exp(self.net.log_q_t_mean-2*torch.exp(self.net.log_q_t_log_sigma))),
            #                                   str(torch.exp(self.net.log_q_t_mean+2*torch.exp(self.net.log_q_t_log_sigma)))))

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
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2+self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def prune(self):
        raise NotImplementedError()
        self.stochastic = False
        self.net.stochastic = False
        self.net.prune()

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def plot_genotype(self, plot_path, caption):
        plot(self.genotype().normal, plot_path+'-normal', caption+'-normal')
        plot(self.genotype().reduce, plot_path+'-reduce', caption+'-reduce')
