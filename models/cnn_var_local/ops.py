""" Operations """
import torch
import torch.nn as nn
import genotypes as gt
import torch.functional as F
import torch as t
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine:
    Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    # 5x5
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    # 9x9
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine)
}
# https://github.com/HolyBayes/pytorch_ard/blob/master/torch_ard/torch_ard.py


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class LocalVarConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stochastic = True
        self.log_sigma = nn.Parameter(
            t.ones(self.weight.shape).to(self.weight.device) * -3.0)
        self.weight.sigma = self.log_sigma
        if self.bias:
            self.log_sigma_b = nn.Parameter(
                t.ones(self.bias.shape).to(self.bias.device) * -3.0)
            self.bias.sigma = self.log_sigma_b

    def forward(self, x):
        if not self.stochastic:
            return nn.functional.conv2d(x, self.weight, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)
        else:
            eps = 1e-8
            W = self.weight
            zeros = torch.zeros_like(W)
            conved_mu = nn.functional.conv2d(x, W, self.bias, self.stride,
                                             self.padding, self.dilation, self.groups)
            log_alpha = self.log_sigma
            # conved_si = torch.sqrt(eps + nn.functional.conv2d(x*x,
            #    torch.exp(log_alpha) * W * W, self.bias, self.stride,
            #    self.padding, self.dilation, self.groups))
            conved_si = torch.sqrt(eps + nn.functional.conv2d(x*x,
                                                              0.01+torch.exp(
                                                                  2*log_alpha), self.bias, self.stride,
                                                              self.padding, self.dilation, self.groups))

            conved = conved_mu + \
                conved_si * \
                torch.normal(torch.zeros_like(conved_mu),
                             torch.ones_like(conved_mu))
            if self.bias:
                conved = conved + (0.01+ torch.exp(2*self.log_sigma_b)) *\
                    torch.normal(torch.zeros_like(conved_mu),
                                 torch.ones_like(conved_mu))
        return conved


class LocalVarLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stochastic = True
        self.log_sigma = nn.Parameter(
            t.ones(self.weight.shape).to(self.weight.device) * -3.0)
        self.weight.sigma = self.log_sigma
        self.log_sigma_b = nn.Parameter(
            t.ones(self.bias.shape).to(self.bias.device) * -3.0)
        self.bias.sigma = self.log_sigma_b

    def forward(self, x):
        if not self.stochastic:
            return nn.functional.linear(x, self.weight, self.bias)
        else:
            W = self.weight
            zeros = torch.zeros_like(W)
            mu = x.matmul(W.t())
            eps = 1e-8
            log_alpha = self.log_sigma
            # si = torch.sqrt((x * x) \
            #                .matmul(((torch.exp(log_alpha) * W * W)+eps).t()))
            si = torch.sqrt((x * x)
                            .matmul(((0.01+torch.exp(2*log_alpha))+eps).t()))
            activation = mu + torch.normal(torch.zeros_like(mu), torch.ones_like(mu)) * si + \
                (0.01+torch.exp(2*self.log_sigma_b)) * \
                torch.normal(torch.zeros_like(mu), torch.ones_like(mu))
            return activation + self.bias


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """

    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(
                kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            LocalVarConv2d(C_in, C_out, kernel_size,
                           stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            LocalVarConv2d(C_in, C_in, (kernel_length, 1),
                           stride, padding, bias=False),
            LocalVarConv2d(C_in, C_out, (1, kernel_length),
                           stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            LocalVarConv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                           bias=False),
            LocalVarConv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        #print (self.net[1])
        self.net[1].forward(x)
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride,
                    padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1,
                    padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = LocalVarConv2d(
            C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = LocalVarConv2d(
            C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class MixedOp(nn.Module):
    """ Mixed operation """

    def __init__(self, C, stride, primitives):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)
        self.log_t = t.zeros(1)
        self.alphas = torch.ones(len(primitives))

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        if len(weights[0].shape) == 1:  # non-scalar
            res = sum(w.view(-1, 1, 1, 1) * op(x)
                      for w, op in zip(weights, self._ops))
        else:
            res = sum(w * op(x) for w, op in zip(weights, self._ops))
        return res
