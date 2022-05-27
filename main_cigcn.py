import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import time
import os
import os.path as osp
import csv
import numpy as np
import yaml

np.random.seed(1337)
#np.random.seed(196)

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

# by gzb: load dataLoader
from feeder.feeder_ntu import NTUDataLoaders, AverageMeter # by gzb: used for sgn model

# by gzb: load models (20211215)
from models.cigcn_debug import CIGCN
from models.cigcn_debug import CascadeModel

# by gzb: load utils
from tools import utils_main
from tools import arg_configs

def load_data(args):
    #ntu_loaders = NTUDataLoaders(args.dataset, args.case, seg=args.seg, args=args)
    ntu_loaders = NTUDataLoaders(args)

    # by gzb: new added code, judge whether exeute rot in dataloader.
    if args.judge_rot == 1:
        train_loader = ntu_loaders.get_train_loader(args.batch_size, args.workers)
    else:
        train_loader = ntu_loaders.gzb_get_train_loader(args.batch_size, args.workers)

    val_loader = ntu_loaders.get_val_loader(args.batch_size, args.workers)
    test_loader = ntu_loaders.get_test_loader(args.batch_size_test, args.workers)

    #'''
    args.train_loader = train_loader
    args.test_loader = test_loader
    args.val_loader = val_loader

class Processor():
    def __init__(self, args):
        self.args = args
        self.network = args.network
        self.milestones = args.milestones

        self.device = args.device

        self.load_model()

        self.load_params()
        self.load_path()

        # save args
        utils_main.save_arg(args, self.save_path)

        #self.load_data()
        if args.judge_inputDataShape == 5:
            self.load_data()
        elif args.judge_inputDataShape == 3:
            self.load_data_sgn()

    def load_model(self):
        if self.args.network == 'CIGCN':
            args.network_label = 0
            model = CIGCN(self.args)
        elif self.args.network == 'CascadeModel':
            args.network_label = 1
            model = CascadeModel(self.args)

        total = utils_main.get_n_params(model)
        #print(model)s# by gzb: 691048 parameters
        print('The number of parameters: ', total)  # by gzb: 691048 parameters
        print('The modes is:', self.args.network)  # by gzb: SGN
        print('device: {}; test_group: {}.'.format(self.args.device, self.args.test_group))
        print('dataSource: {}; dataType: {}; dataMode: {}.'.format(self.args.judge_dataSource, self.args.judge_dataType, self.args.judge_dataMode))

        if torch.cuda.is_available():
            print('It is using GPU!')
            model = model.cuda(self.device)

        self.model = model

    def load_data_sgn(self):
        load_data(self.args)

        self.train_loader = self.args.train_loader
        self.val_loader =self.args.val_loader
        self.test_loader =self.args.test_loader

    def load_params(self):
        self.criterion = utils_main.LabelSmoothingLoss(self.args.num_classes, smoothing=0.1).cuda(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        if self.args.monitor == 'val_acc':
            self.mode = 'max'
            self.monitor_op = np.greater
            self.best = -np.Inf
            self.str_op = 'improve'
        elif self.args.monitor == 'val_loss':
            self.mode = 'min'
            self.monitor_op = np.less
            self.best = np.Inf
            self.str_op = 'reduce'

        self.scheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)

        self.best_epoch = 0
        self.earlystop_cnt = 0
        self.log_res = list()

    def load_path(self):
        #output_dir = utils_main.make_dir(self.args.dataset)  # by gzb: is ./results/NTU/ or ./results/NTU120/
        # 2021-1215
        output_dir = osp.join('./checkpoints', self.args.judge_dataSource) # ./checkpoints/${dataSource}/
        output_dir = osp.join(output_dir, self.args.result_dir) # ./checkpoints/${dataSource}/result_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        save_path = output_dir
        self.save_path = save_path

        dict_format = {
            'ds': self.args.judge_dataSource, # cs or cv for ntu60, csub or cset for ntu120
            'st': self.args.judge_inputDataShape, # 3 for shape (N, 300, 150), 5 for shape (N, C, T, V, M)
            'dt': self.args.judge_dataType, # 0 for joint, 1 for bone
            'net': self.args.network_label,
            'cs': self.args.case, # 0 for cs(csub); 1 for cv(cset)
            'task': self.args.num_classes,
            'rot': self.args.judge_rot, # 0 for without rot, 1 for with rot on data processing
            'seg': self.args.seg, # temp temporal, the num of frames
            'epo': self.args.max_epochs
        } 
        
        '''
        self.checkpoint = osp.join(save_path, 'DS{ds}ST{st}DT{dt}_Net{net}CS{cs}TA{task}_Rot{rot}Seg{seg}Epo{epo}_best.pth'.format(**dict_format))
        self.csv_file = osp.join(save_path, 'DS{ds}ST{st}DT{dt}_Net{net}CS{cs}TA{task}_Rot{rot}Seg{seg}Epo{epo}_log.csv'.format(**dict_format))
        self.lable_path = osp.join(save_path, 'DS{ds}ST{st}DT{dt}_Net{net}CS{cs}TA{task}_Rot{rot}Seg{seg}Epo{epo}_label.txt'.format(**dict_format))
        self.pred_path = osp.join(save_path, 'DS{ds}ST{st}DT{dt}_Net{net}CS{cs}TA{task}_Rot{rot}Seg{seg}Epo{epo}_pred.txt'.format(**dict_format))
        self.print_path = osp.join(save_path, 'DS{ds}ST{st}DT{dt}_Net{net}CS{cs}TA{task}_Rot{rot}Seg{seg}Epo{epo}_print_log.txt'.format(**dict_format))
        '''

        # 20211215
        self.checkpoint = osp.join(save_path, '{ds}_best.pth'.format(**dict_format))
        self.csv_file = osp.join(save_path, '{ds}_log.csv'.format(**dict_format))
        self.lable_path = osp.join(save_path, '{ds}_label.txt'.format(**dict_format))
        self.pred_path = osp.join(save_path, '{ds}_pred.txt'.format(**dict_format))
        self.print_path = osp.join(save_path, '{ds}_print_log.log'.format(**dict_format))

        # 20220104
        os.system('touch {}'.format(self.checkpoint))
        os.system('touch {}'.format(self.print_path))
        os.system('touch {}'.format(self.csv_file))
        os.system('touch {}'.format(self.lable_path))
        os.system('touch {}'.format(self.pred_path))
        
    def train(self, train_loader, model, criterion, optimizer, epoch):
        losses = AverageMeter()
        acces = AverageMeter()
        model.train()

        for i, (inputs, target) in enumerate(train_loader):  # by gzb: shape of input: (batsh_size, seg, 75); (N, C, T, V, M) for HCN, AGCN, etc.
            # by gzb: totally 626 batches, which means i belong to [1, 626]
            # 20211216
            output = model(inputs.cuda(self.device))
            # by gzb: the async is dropped in python3.7
            target = target.cuda(self.device, non_blocking = True)

            '''# 20211231
            with torch.no_grad():
                inputs = inputs.float().cuda(self.device)
                target = target.long().cuda(self.device)
            output = model(inputs)
            '''
            
            if self.args.network == 'CIGCN':
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc = utils_main.accuracy(output.data, target)
                losses.update(loss.item(), inputs.size(0))
                acces.update(acc[0], inputs.size(0))

                # backward
                optimizer.zero_grad()  # clear gradients out before each mini-batch
                loss.backward()
                optimizer.step()


            elif self.args.network == 'CascadeModel':
                '''
                list_loss = 
                list_acc = []
                
                for idx in range(len(output)):
                    #loss.append(criterion(output[idx], target)) # [loss1, loss2, loss3, loss4]
                    list_loss.append(criterion(output[idx], target))
                    list_acc.append(utils_main.accuracy(output.data, target))
                    
                acc = torch.mean(torch.from_numpy)
                acces.update(acc[0], inputs.size(0))
                '''
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc = utils_main.accuracy(output.data, target)
                losses.update(loss.item(), inputs.size(0))
                acces.update(acc[0], inputs.size(0))

                # backward
                optimizer.zero_grad()  # clear gradients out before each mini-batch
                loss.backward()
                optimizer.step()


            if (i + 1) % self.args.print_freq == 0:
                print('Epoch-{:<3d} {:3d} batches\t'
                    'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'accu {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch + 1, i + 1, loss=losses, acc=acces))

                # print log
                with open(self.print_path, 'a') as fout_printLog:
                    print('Epoch-{:<3d} {:3d} batches\t'
                    'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'accu {acc.val:.3f} ({acc.avg:.3f})'.format(epoch + 1, i + 1, loss=losses, acc=acces), file=fout_printLog)               

        return losses.avg, acces.avg

    def validate(self, val_loader, model, criterion):
        losses = AverageMeter()
        acces = AverageMeter()
        model.eval()

        for i, (inputs, target) in enumerate(val_loader):
            with torch.no_grad():
                output = model(inputs.cuda(self.device))
            #target = target.cuda(async=True) # original code
            # by gzb: the async is dropped in python3.7
            target = target.cuda(self.device, non_blocking = True)
            
            with torch.no_grad():
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc = utils_main.accuracy(output.data, target)
            losses.update(loss.item(), inputs.size(0))
            acces.update(acc[0], inputs.size(0))

        return losses.avg, acces.avg

    def test(self, test_loader, model, checkpoint, lable_path, pred_path):
        acces = AverageMeter()
        # load learnt model that obtained best performance on validation set
        model.load_state_dict(torch.load(checkpoint)['state_dict'])
        model.eval()

        label_output = list()
        pred_output = list()

        t_start = time.time()
        for i, (inputs, target) in enumerate(test_loader):
            # by gzb: shape of inputs: (bs * 5, seg, V_C=75)
            # shape of target: (bs, seg, 75)
            with torch.no_grad():
                output = model(inputs.cuda(self.device)) # by gzb: batch_size * 5;  (bs*5, 60) 20211126
                output = output.view((-1, inputs.size(0)//target.size(0), output.size(1))) # (bs, bs*5//bs, 60)
                #output = output.mean(1)
                output = torch.mean(output[:, :self.args.test_group, :], dim=1)

            label_output.append(target.cpu().numpy())
            pred_output.append(output.cpu().numpy())

            #acc = accuracy(output.data, target.cuda(async=True))
            # by gzb:
            acc = utils_main.accuracy(output.data, target.cuda(self.device, non_blocking=True))

            acces.update(acc[0], inputs.size(0))

        label_output = np.concatenate(label_output, axis=0)
        np.savetxt(lable_path, label_output, fmt='%d')
        pred_output = np.concatenate(pred_output, axis=0)
        np.savetxt(pred_path, pred_output, fmt='%f')

        print('Test: accuracy {:.3f}, time: {:.2f}s'
            .format(acces.avg, time.time() - t_start))

        with open(self.print_path, 'a') as fout_printLog:
            print('Test: accuracy {:.3f}, time: {:.2f}s'
                .format(acces.avg, time.time() - t_start), file=fout_printLog)

    def start(self):
        # Training
        if self.args.train == 1:

            # 20211108
            if self.args.resume == 1:
                checkpoint = torch.load(self.checkpoint)
                self.args.start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best = checkpoint['best']

                earlystop_cnt = 0

            for epoch in range(self.args.start_epoch, self.args.max_epochs):

                print(epoch, self.optimizer.param_groups[0]['lr'])

                t_start = time.time()
                train_loss, train_acc = self.train(self.train_loader, self.model, self.criterion, self.optimizer, epoch)
                val_loss, val_acc = self.validate(self.val_loader, self.model, self.criterion)
                self.log_res += [[train_loss, train_acc.cpu().numpy(),\
                            val_loss, val_acc.cpu().numpy()]]

                print('Epoch-{:<3d} {:.1f}s\t'
                    'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                    .format(epoch + 1, time.time() - t_start, train_loss, train_acc, val_loss, val_acc))
                
                with open(self.print_path, 'a') as fout_printLog:
                    print('Epoch-{:<3d} {:.1f}s\t'
                        'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                        .format(epoch + 1, time.time() - t_start, train_loss, train_acc, val_loss, val_acc), file=fout_printLog)

                current = val_loss if self.mode == 'min' else val_acc

                ####### store tensor in cpu
                current = current.cpu()

                if self.monitor_op(current, self.best):
                    print('Epoch %d: %s %sd from %.4f to %.4f, '
                        'saving model to %s'
                        % (epoch + 1, self.args.monitor, self.str_op, self.best, current, self.checkpoint))

                    with open(self.print_path, 'a') as fout_printLog:
                        print('Epoch %d: %s %sd from %.4f to %.4f, '
                        'saving model to %s'
                        % (epoch + 1, self.args.monitor, self.str_op, self.best, current, self.checkpoint), file=fout_printLog)

                    self.best = current
                    best_epoch = epoch + 1

                    utils_main.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'best': self.best,
                        'monitor': self.args.monitor,
                        'optimizer': self.optimizer.state_dict(),
                    }, self.checkpoint)
                    earlystop_cnt = 0
                else:
                    print('Epoch %d: %s did not %s' % (epoch + 1, self.args.monitor, self.str_op))

                    with open(self.print_path, 'a') as fout_printLog:
                        print('Epoch %d: %s did not %s' % (epoch + 1, self.args.monitor, self.str_op), file=fout_printLog)

                    earlystop_cnt += 1

                self.scheduler.step()

            print('Best %s: %.4f from epoch-%d' % (self.args.monitor, self.best, best_epoch))

            with open(self.print_path, 'a') as fout_printLog:
                print('Best %s: %.4f from epoch-%d' % (self.args.monitor, self.best, best_epoch), file=fout_printLog)

            with open(self.csv_file, 'w') as fw:
                cw = csv.writer(fw)
                cw.writerow(['loss', 'acc', 'val_loss', 'val_acc'])
                cw.writerows(self.log_res)
            print('Save train and validation log into into %s' % self.csv_file)

        ### Test
        self.args.train = 0
        self.load_model()

        self.test(self.test_loader, self.model, self.checkpoint, self.lable_path, self.pred_path)

def getConfigArgs():
    parser = arg_configs.get_sgn_parser()
   
    temp_args = parser.parse_args()
 
    if temp_args.config is not None:
        with open(temp_args.config, 'r') as fin_config:
            #fin_config_args = yaml.load(fin_config) # by gzb: return a dict
            fin_config_args = yaml.safe_load(fin_config) # by gzb: return a dict

        arg_file_keys = vars(temp_args).keys()  # by gzb: get all args names from args file
        
        for _key in fin_config_args.keys():
            if _key not in arg_file_keys:
                print('WRONG ARG: {}'.format(_key))
                assert (_key in arg_file_keys)

        parser.set_defaults(**fin_config_args)

    args = parser.parse_args()

    # adaptive judge dataset and case
    if args.judge_dataSource == 'cs' or args.judge_dataSource == 'cv':
        args.dataset = 'NTU'
        args.case = 0 if args.judge_dataSource == 'cs' else 1
    if args.judge_dataSource == 'csub' or args.judge_dataSource == 'cset':
        args.dataset = 'NTU120'
        args.case = 0 if args.judge_dataSource == 'csub' else 1

    #args.num_classes = utils_main.get_num_classes(args.dataset)

    return args   

# for debug of cigcn (binary/multiple classification for partial samples)
def performCigcn(args):
    #args.judge_binary = 1 # for partial samples classification # indicator for data loader

    processor = Processor(args)
    processor.start()

# 20211215
def debugCigcn(args):
    processor = Processor(args)
    processor.start()

    
if __name__ == '__main__':
    
    #''' by gzb: for sgn
    args = getConfigArgs()
    debugCigcn(args)
    '''
    args.train_loader, args.val_loader, args.test_loader = load_data(args)
    processor = sgnProcessor(args)
    processor.start()
    '''


    #performCigcn(args)


    
