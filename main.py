import argparse
import torch
from torch.nn import functional as F
import torch.nn as nn
import os
import numpy as np
import random
from pathlib import Path
import time
# ============================ Data & Func & Networks =====================================
from data.dataset import build_dataset
from util.model_func import train, evaluate
from torchvision import models
from torch.utils.tensorboard import SummaryWriter



# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def build_args():

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=Path(r'/remote/rds/groups/idcom_imaging/data/Face/UTKFace'), help='/gbmormen')
    parser.add_argument('--mode', default='train', help='train|test')
    parser.add_argument('--log_path', default='./log/')
    parser.add_argument('--resume', default=False, type=str, help='path to the lastest checkpoint (default: none)')
    parser.add_argument('--basemodel', default='resnet18', help='resnet18')
    parser.add_argument('--model_name', default='1810whole_UTK', help='mark')
    parser.add_argument('--batch_size', type=int, default=20, help='the mini-batch size of training')
    parser.add_argument('--epochs', type=int, default=60, help='the total number of training epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay for adam. default=0')
    parser.set_defaults(class_weight=None,
                        ckpt_path=None)
    opt = parser.parse_args()

    return opt

def adjust_learning_rate(optimizer, epoch, init_lr=0.001, lr_decay_epoch=10):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs"""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print(f'LR is set to {lr}')

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def model_config(opt):
    if opt.basemodel =='resnet18':
        net = models.resnet18(weights=None, num_classes=10)
    else:
        print(f'Warning: No base model for {opt.basemodel}!!')
    net.to(device)
    optimizer = torch.optim.Adam(
            net.parameters(),
            lr=opt.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=opt.weight_decay
        )
    loss_fn = nn.CrossEntropyLoss(weight=opt.class_weight)

    return net, optimizer, loss_fn

def save_path(opt):
    checkpoint_dir = Path(os.path.join(opt.log_path, opt.model_name, 'checkpoints'))
    log_dir = Path(os.path.join(opt.log_path, opt.model_name, 'logs'))
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    if not checkpoint_dir.exists():
        log_dir.mkdir(parents=True)

    return checkpoint_dir, log_dir



def main():

    opt = build_args()
    print(opt)
    net, optim, loss_fn = model_config(opt)
    checkpoint_dir, log_dir = save_path(opt)
    best_accu = 0

    writer = SummaryWriter(log_dir=log_dir)

    if opt.resume is True:
        ckpt_list = sorted(checkpoint_dir.glob("*.pth"), key=os.path.getmtime)
        if ckpt_list:
            opt.ckpt_path = str(ckpt_list[-1])
        net.load_state_dict(torch.load(opt.ckpt_path))
        print(opt.ckpt_path)
    else:
        print('no checkpoint!!!')
    # ============================ Dataset =====================================
    train_loader, validation_loader, test_loader = build_dataset(root_path=opt.root, batch_size=opt.batch_size, shuffle=True, mode=opt.mode)

    # ============================ Training Stage =====================================
    if opt.mode =='train' or opt.mode =='retain':
        for epoch in range(opt.epochs):

            optim = adjust_learning_rate(optim, epoch, init_lr=opt.lr, lr_decay_epoch=10)
            print('============ training on the train set ============')
            epoch_loss, epoch_acc = train(net, train_loader, opt.root, epoch, device, optim, loss_fn, writer)
            print('============ Validation on the val set ============')
            epoch_val_loss, epoch_val_acc = evaluate(net, validation_loader, opt.root, epoch, device, optim, loss_fn, writer)

            state = net.state_dict()

            is_best = epoch_val_acc > best_accu
            best_accu = max(epoch_val_acc, best_accu)
            writer.add_scalar('best_accu', best_accu, epoch)

            if is_best:
                torch.save(state, checkpoint_dir/f'model_best_epoch_{epoch}.pth')
                print(f'Save checkpoint from epoch {epoch} as best model!!')
            if epoch % 5 == 0:
                torch.save(state, checkpoint_dir/f'checkpoint_{epoch}.pth')
                print(f'Save checkpoint from epoch {epoch} !!')
    # ============================ Testing Stage =====================================
    elif opt.mode =='test':
        for epoch in range(opt.epochs):
            print('============ Testing on the test set ============')
            epoch_test_loss, epoch_test_acc = evaluate(net, test_loader, opt.root, epoch, device, optim, loss_fn, writer)


    writer.close()


if __name__ == '__main__':
    main()