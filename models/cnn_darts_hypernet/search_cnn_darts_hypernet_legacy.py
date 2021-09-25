from ..cnn.search_cnn import SearchCNNController
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.search_cells import SearchCell
from models.cnn_darts_hypernet.architect import HypernetArchitect
from models.cnn import ops
from torch.nn.parallel._functions import Broadcast
from visualize import plot
import genotypes as gt
import logging
import numpy as np


class HyperNet(nn.Module):
    """
    гиперсеть, управляющая нашей структурой
    """
    def __init__(self, hidden_layer_num, hidden_size, out_size1, out_size2):
        """
        :param hidden_layer_num: количество скрытых слоев (может быть нулевым)
        :param hidden_size: количество нейронов на скрытом слое (актуально, если скрытые слои есть)
        :param out_size1: количество строк в матрице, задающей структуру
        :param out_size2: количество столбцов в матрице, задающей структуру
        """
        nn.Module.__init__(self)
        self.out_size1 = out_size1
        self.out_size2 = out_size2
        out_size = out_size1 * out_size2 # выход MLP - вектор, поэтому приводим матрицу к вектору
        layers = []
        in_ = 1 # исходная входная размерность
        for l in range(hidden_layer_num):
            layers.append(nn.Linear(in_, hidden_size))
            layers.append(nn.ReLU())
            in_ = hidden_size
        layers.append(nn.Linear(in_, out_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x --- одномерный вектор (задающий сложность)
        return self.model(x).view(self.out_size1, self.out_size2)


class SearchCNNControllerWithHyperNet(SearchCNNController):
    def __init__(self, **kwargs):
        SearchCNNController.__init__(self, **kwargs)
        self.lam_log_min = float(kwargs['hypernetwork']['log10_lambda_min']) # логарифм минимально допустимой лямбды
        self.lam_log_max = float(kwargs['hypernetwork']['log10_lambda_max']) # логарифм максимально допустимой лямбды
        self.architect = HypernetArchitect(self, float(
            kwargs['darts']['optim']['w_momentum']), float(kwargs['darts']['optim']['w_weight_decay'])) # делаем подмену Architect

    def init_alphas(self,  config):
        primitives = self.get_primitives(config)
        subcfg = config['darts']
        n_layers = int(subcfg['layers'])
        n_nodes = int(subcfg['n_nodes'])

        hypernetwork_hidden_layer_num = int(
            config['hypernetwork']['hidden_layer_num'])
        hypernetwork_hidden_layer_size = int(
            config['hypernetwork']['hidden_layer_size'])

        # initialize architect parameters: alphas
        n_ops = len(primitives)
        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        self.hyper_normal = []
        self.hyper_reduce = []

        # создаем гиперсети
        for i in range(n_nodes):
            if n_layers >= 3:                
                hypernet = HyperNet(
                    hypernetwork_hidden_layer_num, hypernetwork_hidden_layer_size, i+2, n_ops)
                self.alpha_normal.extend(list(hypernet.parameters()))
                self.hyper_normal.append(hypernet)
            hypernet = HyperNet(hypernetwork_hidden_layer_num,
                                hypernetwork_hidden_layer_size, i+2, n_ops)
            self.alpha_reduce.extend(list(hypernet.parameters()))
            self.hyper_reduce.append(hypernet)

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:                
                self._alphas.append((n, p))

    def forward(self, x, lam = None):
        if lam is None:
            # проверка: в обучении forward всегда вызывается с заданной лямбдой. 
            # если лямбда не задана, скорее всего производится оценка качества            
            if self.training:
                raise ValueError('Cannot use default lambda value during training')
            lam = torch.zeros((1,1)).to(self.device)
            
        if self.sampling_mode == 'softmax':
            weights_normal = [F.softmax(alpha(lam)/self.t, dim=-1)
                              for alpha in self.hyper_normal]
            weights_reduce = [F.softmax(alpha(lam)/self.t, dim=-1)
                              for alpha in self.hyper_reduce]

        elif self.sampling_mode == 'naive':
            weights_normal = [alpha(lam) for alpha in self.hyper_normal]
            weights_reduce = [alpha(lam) for alpha in self.hyper_reduce]
        else:
            raise ValueError('Bad sampling mode')

        return self.net(x, weights_normal, weights_reduce)

    def loss(self, X, y, lam):
        logits = self.forward(X, lam)
        return self.criterion(logits, y)

    def hyperloss(self, X, y, lam):
        logits = self.forward(X, lam)
        penalty = 0
        for id, cell in enumerate(self.net.cells):
            # можно не пробегать несколько раз, т.к. клетки одинаковы (С точностью до normal и reduce)            
            weights = [alpha(lam) for alpha in self.hyper_reduce] if cell.reduction else [
                alpha(lam) for alpha in self.hyper_normal]
            for edges, w_list in zip(cell.dag, weights):
                for mixed_op, weights in zip(edges, w_list):
                    for op, w in zip(mixed_op._ops, weights):
                        for param in op.parameters():
                            penalty += torch.norm(param)*w

        return self.criterion(logits, y) + penalty * lam

    def train_step(self, trn_X, trn_y, val_X, val_y):
        # генерация случайной лямбды
        lam = torch.tensor(
            10**np.random.uniform(low=self.lam_log_min, high=self.lam_log_max)).view(1, 1).to(self.device)
        lr = self.lr_scheduler.get_last_lr()[0]
        self.alpha_optim.zero_grad()
        if self.simple_alpha_update:
            arch_loss = self.hyperloss(val_X, val_y, lam)
            arch_loss.backward()
        else:
            self.architect.unrolled_backward(
                trn_X, trn_y, val_X, val_y, lr, self.w_optim, lam)

        self.alpha_optim.step()

        # phase 1. child network step (w)
        self.w_optim.zero_grad()
        loss = self.loss(trn_X, trn_y, lam)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(self.weights(), self.w_grad_clip)

        self.w_optim.step()
        return loss
