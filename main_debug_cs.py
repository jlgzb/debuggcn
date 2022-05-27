# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch

import argparse
import time
import shutil
import os.path as osp
import csv
import numpy as np

np.random.seed(1337)

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
#from model import SGN
from feeder.feeder_ntu import NTUDataLoaders, AverageMeter
import fit
from tools.utils_main import make_dir, get_num_classes, save_arg

#from configs.cigcn import CIGCN
#from configs.cigcn_1119_88968_dv0_89078 import CIGCN
#from configs.cigcn_1119_88920_dv1 import CIGCN
#from configs.cigcn_debug_1130 import CIGCN
#from configs.cigcn_debug_1208 import CIGCN
from models.cigcn_debug import CIGCN

parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
fit.add_fit_args(parser)
parser.set_defaults(
    network='CIGCN',  # by gzb: ori is SGN, which can be HCN, AGCN, AttAGCN, 
    dataset = 'NTU',  # by gzb: NTU for ntu60, NTU120 fro ntu120, see util.py
    case = 0,  # by gzb: 0 for CS setting, 1 for CV setting
    batch_size=64,
    max_epochs=120,
    monitor='val_acc',
    lr=0.001,
    weight_decay=0.0001,
    lr_factor=0.1,
    workers=16, # by gzb: original is 16, but we only have one GPU, so the workers must be 1.
    print_freq = 20,
    train = 1,  # by gzb: 1 for train, and 0 for test
    seg = 20, # by gzb: 20210902, seg means the num of $joint_info (75,) of each sample, shape of $joint_info is: (75,)
    judge_rot=1, # bool, 1 means use rot; 0 means not.
    resume = 0, # by gzb: 1 for resume traing
    judge_dataType = 0, # by gzb: 0 for joint, 1 for bone
    judge_dataMode = 1, # by gzb: 0 for only joint|bone, 1 for t_joint|t_bone, 2 for joint + t_joint | bone + t_bone
    result_dir = 'cigcn_debug_cs_1215_only_t_joint'
    )
args = parser.parse_args()

def main():
    output_dir = make_dir(args.dataset)  # by gzb: is ./results/NTU/ or ./results/NTU120/
    save_path = os.path.join(output_dir, args.network) # by gzb: ./results/NTU/SGN or ./resutls/NTU120/SGN
    # by gzb: new added code
    if args.judge_rot == 1:
        #save_path = osp.join(save_path, 'with_rot_89078_seg20_test64_5')
        #save_path = osp.join(save_path, 'cigcn_debug_cs_1215_only_bone')
        save_path = osp.join(save_path, args.result_dir)
    else:
        save_path = osp.join(save_path, 'without_rot')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.num_classes = get_num_classes(args.dataset)  # by gzb: 60 or 120
    if args.network == 'CIGCN':
        model = CIGCN(args.num_classes, args.dataset, args.seg, args)

    # by gzb: save args
    save_arg(args, save_path)
    #print ('dataset: {}; judge_datatype: {}; mode_data: {}.'.format(args.dataset, args.judge_dataType, args.mode_data))


    total = get_n_params(model)
    #print(model)
    print('The number of parameters: ', total)  # by gzb: 691048 parameters
    print('The modes is:', args.network)  # by gzb: SGN

    if torch.cuda.is_available():
        print('It is using GPU!')
        model = model.cuda()

    criterion = LabelSmoothingLoss(args.num_classes, smoothing=0.1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.monitor == 'val_acc':
        mode = 'max'
        monitor_op = np.greater
        best = -np.Inf
        str_op = 'improve'
    elif args.monitor == 'val_loss':
        mode = 'min'
        monitor_op = np.less
        best = np.Inf
        str_op = 'reduce'

    scheduler = MultiStepLR(optimizer, milestones=[60, 90, 110], gamma=0.1)
    # Data loading
    print ("Start loading dataset...")  # by gzb: new added code
    ntu_loaders = NTUDataLoaders(args.dataset, args.case, seg=args.seg)

    # by gzb: new added code, judge whether exeute rot in dataloader.
    if args.judge_rot == 1:
        train_loader = ntu_loaders.get_train_loader(args.batch_size, args.workers)
    else:
        train_loader = ntu_loaders.gzb_get_train_loader(args.batch_size, args.workers)

    val_loader = ntu_loaders.get_val_loader(args.batch_size, args.workers)
    train_size = ntu_loaders.get_train_size()
    val_size = ntu_loaders.get_val_size()

    print ("GZB: End loading dataset: train and val in main.py !!")  # by gzb

    #test_loader = ntu_loaders.get_test_loader(32, args.workers)
    # by gzb: 20211126
    test_loader = ntu_loaders.get_test_loader(32, args.workers)

    print('Train on %d samples, validate on %d samples' % (train_size, val_size))

    best_epoch = 0
    checkpoint = osp.join(save_path, '%s_best.pth' % args.case)
    earlystop_cnt = 0
    csv_file = osp.join(save_path, '%s_log.csv' % args.case)
    log_res = list()

    lable_path = osp.join(save_path, '%s_lable.txt'% args.case)
    pred_path = osp.join(save_path, '%s_pred.txt' % args.case)

    # 20211115
    print_path = osp.join(save_path, '%s_print_log.txt' % args.case)

    # Training
    if args.train ==1:

        # 20211115 by gzb: for resume train
        if args.resume == 1:
            checkpoint = torch.load(checkpoint)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best = checkpoint['best']

            earlystop_cnt = 0

        for epoch in range(args.start_epoch, args.max_epochs):

            print(epoch, optimizer.param_groups[0]['lr'])

            t_start = time.time()
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
            val_loss, val_acc = validate(val_loader, model, criterion)
            log_res += [[train_loss, train_acc.cpu().numpy(),\
                         val_loss, val_acc.cpu().numpy()]]

            print('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc, val_loss, val_acc))

            # 20211115
            with open(print_path, 'a') as fout_printLog:
                    print('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc, val_loss, val_acc), file=fout_printLog)               

            current = val_loss if mode == 'min' else val_acc

            ####### store tensor in cpu
            current = current.cpu()

            if monitor_op(current, best):
                print('Epoch %d: %s %sd from %.4f to %.4f, '
                      'saving model to %s'
                      % (epoch + 1, args.monitor, str_op, best, current, checkpoint))
                best = current
                best_epoch = epoch + 1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best': best,
                    'monitor': args.monitor,
                    'optimizer': optimizer.state_dict(),
                }, checkpoint)
                earlystop_cnt = 0
            else:
                print('Epoch %d: %s did not %s' % (epoch + 1, args.monitor, str_op))
                earlystop_cnt += 1

            scheduler.step()

        print('Best %s: %.4f from epoch-%d' % (args.monitor, best, best_epoch))
        with open(csv_file, 'w') as fw:
            cw = csv.writer(fw)
            cw.writerow(['loss', 'acc', 'val_loss', 'val_acc'])
            cw.writerows(log_res)
        print('Save train and validation log into into %s' % csv_file)

        with open(print_path, 'a') as fout_printLog:
                    print('Save train and validation log into into %s' % csv_file, file=fout_printLog)

    ### Test
    args.train = 0
    if args.network == 'HCN':
        model = HCN()
    elif args.network == 'SGN':
        model = SGN(args.num_classes, args.dataset, args.seg, args)
    elif args.network == 'CIGCN':
        model = CIGCN(args.num_classes, args.dataset, args.seg, args)

    model = model.cuda()
    test(test_loader, model, checkpoint, lable_path, pred_path)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    acces = AverageMeter()
    model.train()

    for i, (inputs, target) in enumerate(train_loader):  # by gzb: shape of input: (batsh_size, seg, 75)
        # by gzb: totally 626 batches, which means i belong to [1, 626]
        output = model(inputs.cuda())
        #target = target.cuda(async = True)

        # by gzb: the async is dropped in python3.7
        target = target.cuda(non_blocking = True)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # backward
        optimizer.zero_grad()  # clear gradients out before each mini-batch
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print('Epoch-{:<3d} {:3d} batches\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accu {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch + 1, i + 1, loss=losses, acc=acces))

    return losses.avg, acces.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()

    for i, (inputs, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model(inputs.cuda())
        #target = target.cuda(async=True) # original code
        # by gzb: the async is dropped in python3.7
        target = target.cuda(non_blocking = True)
        
        with torch.no_grad():
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

    return losses.avg, acces.avg


def test(test_loader, model, checkpoint, lable_path, pred_path):
    acces = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()

    label_output = list()
    pred_output = list()

    t_start = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            output = model(inputs.cuda())
            output = output.view((-1, inputs.size(0)//target.size(0), output.size(1)))
            #output = output.mean(1)
            output = torch.mean(output[:, :10, :], dim=1)


        label_output.append(target.cpu().numpy())
        pred_output.append(output.cpu().numpy())

        #acc = accuracy(output.data, target.cuda(async=True))
        # by gzb:
        acc = accuracy(output.data, target.cuda(non_blocking=True))

        acces.update(acc[0], inputs.size(0))


    label_output = np.concatenate(label_output, axis=0)
    np.savetxt(lable_path, label_output, fmt='%d')
    pred_output = np.concatenate(pred_output, axis=0)
    np.savetxt(pred_path, pred_output, fmt='%f')

    print('Test: accuracy {:.3f}, time: {:.2f}s'
          .format(acces.avg, time.time() - t_start))


def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)

    return correct.mul_(100.0 / batch_size)

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

if __name__ == '__main__':


    main()

    # by gzb: for ntu120: The number of parameters:  721828
    # by gzb: for ntu60: The number of parameters: 691048
    
