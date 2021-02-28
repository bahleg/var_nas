""" CNN for architecture search """
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




def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class SearchCNN(nn.Module):
    """ Search CNN model """

    def __init__(self, primitives,  C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
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
                              reduction_p, reduction, primitives)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, **kwargs):
        super().__init__()        
        subcfg = kwargs['darts']
        C_in = int(subcfg['input_channels'])
        C = int(subcfg['init_channels'])
        n_classes = int(subcfg['n_classes'])
        n_layers = int(subcfg['layers'])
        n_nodes = int(subcfg['n_nodes'])
        stem_multiplier = int(subcfg['stem_multiplier'])     
        self.sampling_mode = subcfg['sampling_mode']
        self.n_nodes = n_nodes        
        self.device = kwargs['device']
        self.criterion = nn.CrossEntropyLoss().to(self.device)        
        self.t = float(subcfg['initial temp'])
        self.init_t = float(subcfg['initial temp'])
        
        self.delta_t = float(subcfg['delta'])
        # initialize architect parameters: alphas
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

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()
        
        if self.sampling_mode == 'igr':
            self.alpha_cov_normal = nn.ParameterList()
            self.alpha_cov_reduce = nn.ParameterList()

        for i in range(n_nodes):
            if n_layers>=3:
                self.alpha_normal.append(                    
                    nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
                if self.sampling_mode == 'igr':                
                    self.alpha_cov_normal.append(nn.Parameter(torch.randn(i+2, n_ops, n_ops)*0.1))
            self.alpha_reduce.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            if self.sampling_mode == 'igr':                
                self.alpha_cov_reduce.append(nn.Parameter(torch.randn(i+2, n_ops, n_ops)*0.1))


        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(primitives, C_in, C, n_classes, n_layers,
                             n_nodes, stem_multiplier )
        
        # weights optimizer
        self.w_optim = torch.optim.SGD(self.weights(), float(subcfg['optim']['w_lr']), momentum=float(subcfg['optim']['w_momentum']),
                                weight_decay=float(subcfg['optim']['w_weight_decay']))
        # alphas optimizer
        self.alpha_optim = torch.optim.Adam(self.alphas(), float(subcfg['optim']['alpha_lr']), betas=(0.5, 0.999),
                                    weight_decay=float(subcfg['optim']['alpha_weight_decay']))

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.w_optim, int(kwargs['epochs']), eta_min=float(subcfg['optim']['w_lr_min']))

        self.simple_alpha_update = int(subcfg['optim']['simple_alpha_update'])!=0
        self.architect = Architect(self, float(subcfg['optim']['w_momentum']), float(subcfg['optim']['w_weight_decay']))
        self.w_grad_clip = float(subcfg['optim']['w_grad_clip'])

    def train_step(self, trn_X, trn_y, val_X, val_y):
        lr = self.lr_scheduler.get_last_lr()[0]
        self.alpha_optim.zero_grad()
        if self.simple_alpha_update:
            arch_loss = self.architect.net.loss(val_X, val_y)
            arch_loss.backward()            
        else:
            self.architect.unrolled_backward(
                trn_X, trn_y, val_X, val_y, lr, self.w_optim)
        
        self.alpha_optim.step()
      

        # phase 1. child network step (w)
        self.w_optim.zero_grad()
        loss = self.loss(trn_X, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(self.weights(), self.w_grad_clip)

        self.w_optim.step()        
        return loss 

        

    def forward(self, x):
        if self.sampling_mode == 'softmax':
            weights_normal = [F.softmax(alpha/self.t, dim=-1)
                                      for alpha in self.alpha_normal]
            weights_reduce = [F.softmax(alpha/self.t, dim=-1)
                                    for alpha in self.alpha_reduce]

        elif self.sampling_mode == 'gumbel-softmax':
            weights_normal = [torch.distributions.RelaxedOneHotCategorical(
                    self.t, logits=alpha).rsample([x.shape[0]]) for alpha in self.alpha_normal]
            weights_reduce = [torch.distributions.RelaxedOneHotCategorical(
                    self.t, logits=alpha).rsample([x.shape[0]]) for alpha in self.alpha_reduce]        
        elif self.sampling_mode == 'naive':
            weights_normal = self.alpha_normal 
            weights_reduce = self.alpha_reduce     
        elif self.sampling_mode == 'igr':

            weights_normal = []
            weights_reduce = []

            for alpha, cov in zip(self.alpha_normal, self.alpha_cov_normal):  
                subsample = []             
                for subalpha, subcov in zip(alpha, cov):                    
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
        return self.criterion(logits, y)

    def weights(self):
        return self.net.parameters()


    def alphas(self):
        for n, p in self._alphas:
            yield p

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
        
        if self.sampling_mode == 'igr':
            logger.info("\n# Covariance - normal")
            for alpha in self.alpha_cov_normal:
                logger.info(alpha)
            logger.info("\n# Covariance - reduce")
            for alpha in self.alpha_cov_reduce:
                logger.info(alpha)        

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

    def plot_genotype(self, plot_path, caption):
        plot(self.genotype().normal, plot_path+'-normal', caption+'-normal')
        plot(self.genotype().reduce, plot_path+'-reduce', caption+'-reduce')

    def new_epoch(self, e, w, l):
        self.lr_scheduler.step(epoch=e)    
        self.t = self.init_t + self.delta_t*e
        #self.print_alphas(l)        

    def writer_callback(self, writer,  epoch, cur_step):
        pass
        #hist_values = []
        #for val in self.alphas():
        #    hist_values.extend(F.softmax(val).cpu().detach().numpy().tolist())
        #hist_values = np.array(hist_values).flatten().tolist()
        #writer.add_histogram('train/alphas', hist_values, cur_step)  # %%

        #writer.add_scalar('train/temp_min', torch.exp(self.net.log_q_t_mean-2*torch.exp(self.net.log_q_t_log_sigma)).cpu().detach().numpy(), cur_step)
        #writer.add_scalar('train/temp_max', torch.exp(self.net.log_q_t_mean+2*torch.exp(self.net.log_q_t_log_sigma)).cpu().detach().numpy(), cur_step)
