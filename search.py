""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
import sys
from tensorboardX import SummaryWriter
import utils
from importlib import import_module
from utils import Config

def main():    
    config = Config(sys.argv[1])
    device = torch.device("cuda")

    # tensorboard
    tb_path = os.path.join(config.path, "tb")
    os.system('rm -r '+tb_path)
    writer = SummaryWriter(log_dir=tb_path)
    writer.add_text('config', str(config.__dict__), 0)

    logger = utils.get_logger(os.path.join(
        config.path, "{}_train.log".format(config.name)))    

    logger.info("Logger is set - training start")
    torch.cuda.set_device(config.device)

    # set seed
    

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)

    module_name, class_name = config.model_class.rsplit('.', 1)
    controller_cls = getattr(import_module(module_name), class_name)
    model = None 
    for seed in config.seeds.split(';'):    
        seed = int(seed)
        logger.info('using seed '+str(seed))
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if model is not None:
            del model 
            if 'cuda' in config.device:
                torch.cuda.empty_cache() 

        model = controller_cls(**config.__dict__)
        model = model.to(device)    
        # split data to train/validation
        n_train = len(train_data)
        split = int(n_train * float(config.validate_split))
        indices = list(range(n_train))
        if split <= 0:
            logger.info('using train as validation')
            valid_sampler = train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices)
        else:
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[:split])
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:])

        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=int(config.batch_size),
                                                sampler=train_sampler,
                                                num_workers=int(config.workers),
                                                pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=int(config.batch_size),
                                                sampler=valid_sampler,
                                                num_workers=int(config.workers),
                                                pin_memory=True)

        

        # training loop
        best = 0    
        
        for epoch in range(int(config.epochs)):
            
            model.new_epoch(epoch, writer, logger) 
            # training
            train_qual = train(train_loader, valid_loader, model, epoch, writer,  config, logger)        
                        
            # validation
            cur_step = (epoch+1) * len(train_loader)
            val_qual = validate(valid_loader, model, epoch,
                                cur_step, device, config, logger, writer)
            plot_path = os.path.join(config.plot_path, "EP{:02d}_Seed{}".format(epoch+1, seed))                            
            caption = "Epoch {}".format(epoch+1)
            model.plot_genotype(plot_path, caption)
            
            if int(config.use_train_quality) != 0:
                cur_qual = train_qual
            else:
                cur_qual = val_qual

            # save
            if best < cur_qual:
                best = cur_qual            
                is_best = True
            else:
                is_best = False
            utils.save_checkpoint(model, config.path, '_seed_'+str(seed), is_best)
            logger.info("Quality{}: {} \n\n".format(
                '*' if is_best else '', cur_qual))

        logger.info("Final best =  {}".format(best))    


def train(train_loader, valid_loader, model, epoch, writer,  config, logger):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(config.device, non_blocking=True), trn_y.to(
            config.device, non_blocking=True)
        val_X, val_y = val_X.to(config.device, non_blocking=True), val_y.to(
            config.device, non_blocking=True)
        N = trn_X.size(0)

        loss = model.train_step(trn_X, trn_y, val_X, val_y)
        losses.update(loss.item(), N)

        if step ==  len(train_loader)-1:
            logits = model(trn_X)
            prec1 = utils.accuracy(
                logits, trn_y)[0]
            top1.update(prec1.item(), N)
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Acc@(1) {top1.avg:.1%}".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1))
            model.writer_callback(writer, epoch, cur_step)
        
        
            writer.add_scalar('train/loss', loss.item(), cur_step)
            writer.add_scalar('train/top1', prec1.item(), cur_step)
            
        cur_step += 1

    logger.info(
        "Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    if config.quality == 'negloss':
        return -losses.avg
    elif config.quality == 'top1':
        return top1.avg
    elif config.quality == 'last':
        return cur_step


def validate(valid_loader, model, epoch, cur_step, device, config, logger, writer):
    top1 = utils.AverageMeter()    
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(
                device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)
            prec1 = utils.accuracy(
                logits, y)[0]

            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
                            

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    

    logger.info(
        "Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    if config.quality == 'negloss':
        return -losses.avg
    elif config.quality == 'top1':
        return top1.avg
    elif config.quality == 'last':
        return cur_step


if __name__ == "__main__":
    main()
