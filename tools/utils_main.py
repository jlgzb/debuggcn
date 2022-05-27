#from feeder.feeder import NTUDataset
#from main import train
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import shutil
import os
import yaml
#from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
import time
import random
import inspect
import pickle
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm

import sys
import yaml

def make_dir(dataset):
    if dataset == 'NTU':
        output_dir = os.path.join('./results/NTU/')
    elif dataset == 'NTU120':
        output_dir = os.path.join('./results/NTU120/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def get_num_classes(dataset):
    if dataset == 'NTU':
        return 60
    elif dataset == 'NTU120':
        return 120

def save_arg(args, save_path):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open('{}/config.yaml'.format(save_path), 'w') as f:
        f.write(f"# command line: {' '.join(sys.argv)}\n\n")
        yaml.dump(arg_dict, f)

#''' # load data from npy file
def getSampleNum(args):
    train_label_path = args.train_feeder_args['label_path']
    test_label_path = args.test_feeder_args['label_path']

    dict_train_sample_num = {}
    dict_test_sample_num = {}

    for _idx in range(120):
        dict_train_sample_num[str(_idx)] = 0
    
    for _idx in range(120):
        dict_test_sample_num[str(_idx)] = 0

    # load label
    with open(train_label_path, 'rb') as fin_train:
        train_sample_name, train_Y = pickle.load(fin_train)
    with open(test_label_path, 'rb') as fin_test:
        test_sample_name, test_Y = pickle.load(fin_test)

    print ("....train_y value: {}".format(max(train_Y)))

    for _label in train_Y:
        dict_train_sample_num[str(_label)] = dict_train_sample_num[str(_label)] + 1
    for _label in test_Y:
        dict_test_sample_num[str(_label)] = dict_test_sample_num[str(_label)] + 1

    list_trainSamplesNum = []
    list_testSamplesNum = []

    fout_tr = open('./data_samples/sampleNum4eachAction_train.txt', 'w')
    for _key in dict_train_sample_num.keys():
        _value = dict_train_sample_num[_key]
        print (_key,',', _value, file=fout_tr)

        list_trainSamplesNum.append(_value)

    fout_tr.close()

    fout_te = open('./data_samples/sampleNum4eachAction_test.txt', 'w')
    for _key in dict_test_sample_num.keys():
        _value = dict_test_sample_num[_key]
        print (_key, ',', _value, file=fout_te)

        list_testSamplesNum.append(_value)

    fout_te.close

    print ("....train and test samples sum: tr{} te{}".format(sum(list_trainSamplesNum), sum(list_testSamplesNum)))

# extract labelIndex by labels
def getLabelIndex(label_path, list_labelIdx):
    # load label from file
    with open(label_path, 'rb') as fin:
        sample_name, label_Y = pickle.load(fin)

    list_labelIndex = []
    # extract binary idx, label in label_Y is 0-based (0-59)
    for _idx, _label in enumerate(label_Y):
        if _label in list_labelIdx:
            list_labelIndex.append(_idx)  # record the index of target label

    # extract labels which we want
    label_Y = np.array(label_Y)[list_labelIndex]
    
    return label_Y, list_labelIndex

def renameBinaryLabel(label_Y): # type of label_Y is np.ndarray
    # 14, 15, 16, 17, 18, 19, 20, 21
    _label_positive = 19
    _label_negative = 20

    list_label_Y = []
    for _idx, _label in enumerate(label_Y):
        if _label == _label_positive:
            list_label_Y.append(1)
        elif _label == _label_negative:
            list_label_Y.append(0)

    return list_label_Y

def renameMultiLabel(label_Y):
    list_labelID_0 = [13, 14]
    list_labelID_1 = [15, 16]
    list_labelID_2 = [17, 18]
    list_labelID_3 = [19, 20]

    list_label_Y = []
    for _idx, _label in enumerate(label_Y):
        if _label in list_labelID_0:
            list_label_Y.append(0)
        elif _label in list_labelID_1:
            list_label_Y.append(1)
        elif _label in list_labelID_2:
            list_label_Y.append(2)
        elif _label in list_labelID_3:
            list_label_Y.append(3)

    return list_label_Y


def create_dataset_full(args):
    train_data_path = args.train_feeder_args['data_path'] # by gzb: cross sub
    train_label_path = args.train_feeder_args['label_path']
    test_data_path = args.test_feeder_args['data_path']
    test_label_path = args.test_feeder_args['label_path']

    train_X = np.load(train_data_path) # by gzb: return shape: (N, C, T, V, M)
    test_X = np.load(test_data_path) # by gzb: type: np.ndarray

    with open(train_label_path, 'rb') as fin:
        sample_name, train_Y = pickle.load(fin)
    with open(test_label_path, 'rb') as fin:
        sample_name, test_Y = pickle.load(fin)
    
    #train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.05, random_state=10000)

    #return train_X, train_Y, val_X, val_Y, test_X, test_Y
    return train_X, train_Y, test_X, test_Y

def create_dataset_JntBone(args):
    train_jnt_path = args.train_feeder_paths['joint_path']
    train_bone_path = args.train_feeder_paths['bone_path']
    train_label_path = args.train_feeder_paths['label_path']
    test_jnt_path = args.test_feeder_paths['joint_path']
    test_bone_path = args.test_feeder_paths['bone_path']
    test_label_path = args.test_feeder_paths['label_path']

    train_X_jnt = np.load(train_jnt_path) # by gzb: return shape: (N, C, T, V, M)
    train_X_bone = np.load(train_bone_path) # by gzb: return shape: (N, C, T, V, M)
    test_X_jnt = np.load(test_jnt_path)
    test_X_bone = np.load(test_bone_path)

    with open(train_label_path, 'rb') as fin:
        sample_name, train_Y = pickle.load(fin)
    with open(test_label_path, 'rb') as fin:
        sample_name, test_Y = pickle.load(fin)

    train_X_jnt = np.concatenate((train_X_jnt, train_X_bone), axis=0) # (N*2, C, T, V, M)
    #test_X_jnt = np.concatenate((test_X_jnt, test_X_bone), axis=0)

    train_Y = np.concatenate((train_Y, train_Y), axis=0)
    #test_Y = np.concatenate((test_Y, test_Y), axis=0)

    #train_X_jnt, val_X, train_Y, val_Y = train_test_split(train_X_jnt, train_Y, test_size=0.05, random_state=10000)
    #return train_X_jnt, train_Y, val_X, val_Y, test_X_jnt, test_Y

    return train_X_jnt, train_Y, test_X_jnt, test_Y



def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

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

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            true_dist = torch.zeros_like(pred).cuda(pred.get_device())
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# from 2s-agcn
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

'''
# not used
class agcnProcessor():
    """ 
        Processor for Skeleton-based Action Recgnition
        by gzb: refer from main.py of 2s-agcn
    """
    def __init__(self, arg):
        self.arg = arg
        self.phase = self.arg.phase
        self.debug = arg.train_feeder_args['debug']
        self.save_arg()
        
        if self.phase == 'train' and self.debug == False:
            if os.path.isdir(arg.model_saved_name):
                # by gzb: remove the exist save dir
                print('log_dir: ', arg.model_saved_name, 'already exist')
                answer = input('delete it? y/n:')
                if answer == 'y':
                    shutil.rmtree(arg.model_saved_name)
                    print('Dir removed: ', arg.model_saved_name)
                    input('Refresh the website of tensorboard by pressing any keys')
                else:
                    print('Dir not removed: ', arg.model_saved_name)
            self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
            self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
        
        elif self.phase == 'train' and self.debug == True:
            self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')

        self.load_data()
        self.load_model()
        self.load_optimizer()

        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        NTUDataset = import_class(self.arg.feeder)

        self.data_loader = dict()
        if self.phase == 'train':

            # by gzb: get train loader
            train_loader = DataLoader(
            dataset = NTUDataset(**self.arg.train_feeder_args),
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker,
            drop_last=True,
            worker_init_fn=init_seed)

            self.data_loader['train'] = train_loader

        # by gzb: get test loader
        test_loader = DataLoader(
            dataset = NTUDataset(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed
        )
        self.data_loader['test'] = test_loader


    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device

        model = import_class(self.arg.model)
        #shutil.copy2(inspect.getfile(model), self.arg.work_dir)

        print(model)
        #self.model = model(**self.arg.model_args).cuda(output_device)
        self.model = model().cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        # by gzb: not used
        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            #weights = OrderedDict(
            #    [[k.split('module.')[-1],
            #      v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        # by gzb: not used for only one devices
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))


    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)  # by gzb: return a dict
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str) # by gzb: print message into console

        # by gzb: save log into ~/log.txt
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        # print(key + '-require grad')
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                        # print(key + '-not require grad')
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # forward
            output = self.model(data)
            # if batch_idx == 0 and epoch == 0:
            #     self.train_writer.add_graph(self.model, output)
            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0
            loss = self.loss(output, label) + l1

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_l1', l1, self.global_step)
            # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #     self.print_log(
            #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #             batch_idx, len(loader), loss.data[0], lr))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        if save_model:
            state_dict = self.model.state_dict()
            #weights = OrderedDict([[k.split('module.')[-1],
            #                        v.cpu()] for k, v in state_dict.items()])
            weights = 0 # by gzb: not find above OrderedDict function

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = Variable(
                        data.float().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)
                    label = Variable(
                        label.long().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)
                    output = self.model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
            # self.lr_scheduler.step(loss)
            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):
        if self.phase == 'train':
            self.print_log(  # by gzb: print parameters fo arg
                'Parameters:\n{}\n'.format(str(vars(self.arg)))
            )
            
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-3:
                    break

                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)

                self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])

            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)

        elif self.phase == 'test':
            if self.debug == False:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')
'''



