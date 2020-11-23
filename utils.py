""" Utilities """
import os
import logging
import shutil
import glob

import torch
import torchvision.datasets as dset
import numpy as np
import preproc
import torch as t
from configobj import ConfigObj



          

def get_data(dataset, data_path, cutout_length, validation):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10        
    elif dataset  == 'toy':
         dset_cls = DsetMock
         n_classes = 2
    else:    
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    try:
            shape = trn_data.train_data.shape
    except:
            shape = trn_data.data.shape
    input_channels = 3 if len(shape) == 4 else 1
    #assert d shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation: # append validation data
        ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))

    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    os.system('rm '+file_path)
    
    logging.getLogger().handlers = []
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    
    
    return logger



class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res

def find_checkpoint(ckpt_dir, seed):
    files = sorted(glob.glob(ckpt_dir+'/'+'checkpoint_{}_*.ckp'.format(seed)), key = lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(files)>0:
        return files[-1], int(files[-1].split('_')[-1].split('.')[0]) + 1
    else:
        return None, 0
    

def save_checkpoint(state, ckpt_dir, seed='', epoch='',  is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint_{}_{}.ckp'.format(seed, epoch))
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best_{}.pth.tar'.format(seed))
        shutil.copyfile(filename, best_filename)


class Config:
    def __init__(self, config_path):            
        self.name = 'experiment'        
        self.workers = 4         
        self.path = os.path.join('searchs', self.name)
        self.plot_path = self.path
        self.device = 'cuda:0'
        self.data_path = './data'        
        self.cfg = ConfigObj(config_path)        
        for k in self.cfg:
            self.__dict__[k] = self.cfg[k]