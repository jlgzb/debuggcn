from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os.path as osp
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class NTUDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = np.array(y, dtype='int')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return [self.x[index], int(self.y[index])]

class NTUDataLoaders(object):
    def __init__(self):

        self.seg = 20

        ''' by gzb: 
        create dataset. Generate following variable:
            self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y
            but self.val_X == self.test_X; self.val_Y == self.test_Y
        '''
        self.create_datasets_from_npz()
        
        self.train_set = NTUDataset(self.train_X, self.train_Y)  # by gzb: create train_set
        print ("GZB: Complete torch.Dataset for train_set, shape is: {} !".format(self.train_X.shape))

        self.val_set = NTUDataset(self.val_X, self.val_Y)
        print ("GZB: Complete torch.Dataset for val_set, shape is: {} !".format(self.val_X.shape))

        self.test_set = NTUDataset(self.test_X, self.test_Y)
        print ("GZB: Complete torch.Dataset for test_set, shape is: {} !".format(self.test_X.shape))

    def get_train_loader(self, batch_size, num_workers):
        # by gzb: drop_last: if the samples in last batch is not equal to batch_size, drop this batch for True.
        return DataLoader(self.train_set, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            collate_fn=self.collate_fn_fix_train, pin_memory=True, drop_last=True)

    def get_val_loader(self, batch_size, num_workers):
        return DataLoader(self.val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=self.collate_fn_fix_val, pin_memory=True, drop_last=True)


    def get_test_loader(self, batch_size, num_workers):
        return DataLoader(self.test_set, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          collate_fn=self.collate_fn_fix_test, pin_memory=True, drop_last=True)

    def get_train_size(self):
        return len(self.train_Y)  # by gzb: shape of train_Y is (train_sample_num + val_sample_num, )

    def get_val_size(self):
        return len(self.val_Y)

    def get_test_size(self):
        return len(self.test_Y)

    def create_datasets_from_npz(self):
        print ("GZB: Load data from CS npz file in feeder_ntu.py...")  # by gzb: now added code.

        path = osp.join('./data/', 'NTU_CS.npz')
        npz_data = np.load(path)

        self.train_X = npz_data['x_train']
        self.train_Y = np.where(npz_data['y_train'] > 0)[1]

        self.test_X = npz_data['x_test']
        self.test_Y = np.where(npz_data['y_test'] > 0)[1]

        self.val_X = self.test_X
        self.val_Y = self.test_Y

        print ("GZB: End load data from CS npz file in feeder_ntu.py !!")  # by gzb: now added code.

    # by gzb: define the way that how to extract data from Dataset
    def collate_fn_fix_train(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        x, y = zip(*batch)

        x, y = self.Tolist_fix(x, y, train=1)  # by gzb: shape of x: (sample_num, self.seg, 75), sample_num may be batch_size

        # by gzb: lens is a list: [20, 20, ...] (20210902), shape of lens: (sample_num,)
        # by gzb: the following two code line: 
        # step 1: restore the frame_num (num of actor skeleton) of each sample in to "lens" (a list); 20210902
        # step 2: return the index of lens by value in descending order. if self.seg==20, these two steps is meaningless
        lens = np.array([x_.shape[0] for x_ in x], dtype=np.int64)  # by gzb: x_ is joint_info of one sample, s.t. x_.shape[0] is $self.seg (20210902).
        idx = lens.argsort()[::-1]  # sort sequence by valid length in descending order  # by gzb: sort value by descending order, and return the index list.
        
        y = np.array(y)[idx]  # by gzb: 
        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)  # by gzb: Translate type of x from numpy to torch. shape: (idx, self.seq, 75) (20210902). 

        theta = 0.3

        #### data augmentation
        # by gzb: execute 3D rotaion, refer to ntu rgb+d60 paper
        # by gzb: to fix the X axis parallel to the 3D vector from "right shoulder" to "left shoulder"
        # by gzb: to fix the Y axis towards the 3D vector from "spine base" to "spine" (middle of the spine)
        # by gzb: the Z axis is fixed as the new X x Y
        # by gzb: in the nomalization step, to scale all the 3D points based on the distance between "spine base" and "spine" joints.

        x = _transform(x, theta) 
        #### data augmentation
        y = torch.LongTensor(y)

        return [x, y]

    def collate_fn_fix_val(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        x, y = zip(*batch)
        x, y = self.Tolist_fix(x, y, train=1)
        idx = range(len(x))
        y = np.array(y)

        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        y = torch.LongTensor(y)

        return [x, y]

    def collate_fn_fix_test(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        x, y = zip(*batch)
        x, labels = self.Tolist_fix(x, y, train=2)
        idx = range(len(x))
        y = np.array(y)

        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        y = torch.LongTensor(y)

        return [x, y]

    def Tolist_fix(self, joints, y, train = 1):  # by gzb: joints is data, y is label. shape of joint is (batch_size, 300, 150)?
        # by gzb: process of this func
        # step 1: delete frames of each sample of which joint info are all zeros
        # step 2: split joint of each frame into two, from (150,) to (75,), and restore into new list ("seq") for each sample
        # step 3: random extract $self.seg joint info(75,) from "seq" for each sample, and update seq of each sample. shape of seq: (20, 75)
        # step 4: resotre "seq" into "seqs", shape of seqs: (sample_num, self.seg=20, 75) (20210902)
        seqs = []

        # by gzb: seq means the info of sample, shape: (frame_num, joint_vector)
        for idx, seq in enumerate(joints):  # by gzb:, idx is index of sample; shape of seq is: (300, 150)
            zero_row = [] # record indexID of useless frame
            for i in range(len(seq)):    # by gzb: len(seq) means the frame_num of each sample
                # by gzb: if joint info of one frame are all zero.
                if (seq[i, :] == np.zeros((1, 150))).all():  # by gzb: seq[i, :] means the joint info of each frame
                        zero_row.append(i)

            # by gzb: after the following code, len(seq) is changed.
            seq = np.delete(seq, zero_row, axis = 0)  # by gzb: delete frames of seq, which are useless frames, the joint info of which are all zero.

            # by gzb: len(seq) is not equal to 'y', seq is the joint info of one frame.
            seq = turn_two_to_one(seq)  # by gzb: seq is a list which restore the joint info based on one actor, e.g. [(75,) , (75,), ...], shape: (-1, 75)
            seqs = self.sub_seq(seqs, seq, train=train)  # by gzb: shape of seqs: (-1, self.seg, 75). YES (20210902) !

        return seqs, y  # by gzb: shape of seqs is: (sample_num, self.seg=20, 75)

    # by gzb: do what ? 20210902: Reason: extract joint info by skipping frames (skipping actor now)
    def sub_seq(self, seqs, seq, train = 1):
        group = self.seg  # by gzb: is 20 on training

        # by gzb: shape of seq is (-1, 75). If the frames num < self.seq, e.g. 11(thres in denoised script) < 20
        # by gzb: 20210902, now seq is restore joint info of one actor (-1, 75)
        if seq.shape[0] < self.seg: # by gzb: here len(seq) is not num_frame after "turn_two_to_one" func; actualy is num of useful actor info of all frames of one sample (joint info are not zeros)
            pad = np.zeros((self.seg - seq.shape[0], seq.shape[1])).astype(np.float32)  # by gzb: shape of pad is (self.seg - seq.shape[0], 75), with zero values
            seq = np.concatenate([seq, pad], axis=0)  # by gzb: add zeros info into seq. 20210902: make shape of seq is : (20, 75)

        # by gzb: interval of skip actors (20210902).
        ave_duration = seq.shape[0] // group  # by gzb 20210902: if seq.shape[0]==20, s.t. ave_duration is 1

        if train == 1:  # by gzb: list(range(group)) is [0, 1, ..., 20]
            # by gzb: offsets: create random list which shape (group, ), func is: a * b + c (a is a list, b is a int num, c is a list)
            offsets = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group) # by gzb 20210902: [0, 1, ...19] if ave_duration == 1
            seq = seq[offsets]  # by gzb: offsets is random index.  shape of seq is (sefl.seg, 75) after this code line
            seqs.append(seq)  # by gzb: random add $self.seg joint info to restore into seqs; shape of seqs is (-1, self.seg, 75) 

        elif train == 2:  # by gzb: for test

            for idx in range(20):
                offsets = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
                seqs.append(seq[offsets])



        return seqs # by gzb: extract joint_info 5 times by skipping frames (actor)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# by gzb: only record unzero joint data, tern one to two? (20210902, Reason: turn two actor to one actor, means extract joint info in terms of actor )
def turn_two_to_one(seq):
    new_seq = list()  # by gzb: which is used to restore unzero joint info, since the joint info may be all zero after "seq_transformation.py"
    for idx, ske in enumerate(seq):  # by gzb: ske is the info of joint, whic shape (150,), shape of seq is (-1, 150)
        if (ske[0:75] == np.zeros((1, 75))).all():
            new_seq.append(ske[75:])  # by gzb: ske[75:] can not be zero, since [0:150]==0 is already deleted
        elif (ske[75:] == np.zeros((1, 75))).all():
            new_seq.append(ske[0:75])
        else:
            # all value of joint position are not zero, actor1 and actor2
            new_seq.append(ske[0:75])
            new_seq.append(ske[75:])
    return np.array(new_seq)


# by gzb: Not fully understood (20210902)
def _rot(rot):  # by gzb: shape of rot: (sample_num, self.seg, 3)
    cos_r, sin_r = rot.cos(), rot.sin()  # by gzb: execute cos or sin on values of "rot" tensor
    zeros = rot.new(rot.size()[:2] + (1,)).zero_()  # by gzb: zero-value tensor with shape(sample_num, self.seg, 1)
    ones = rot.new(rot.size()[:2] + (1,)).fill_(1)  # by gzb: one-value tensor with shape(sample_num, self.seg, 1)

    r1 = torch.stack((ones, zeros, zeros), dim=-1)  # dim=-1 means: shape of r1: (sample_num, self.seg, 1, 3)
    rx2 = torch.stack((zeros, cos_r[:,:,0:1], sin_r[:,:,0:1]), dim = -1)
    rx3 = torch.stack((zeros, -sin_r[:,:,0:1], cos_r[:,:,0:1]), dim = -1)
    rx = torch.cat((r1, rx2, rx3), dim = 2) # by gzb: shape of rx: (sample_num, self.seg, 1*3 (since dim==2), 3)

    ry1 = torch.stack((cos_r[:,:,1:2], zeros, -sin_r[:,:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:,:,1:2], zeros, cos_r[:,:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 2)

    rz1 = torch.stack((cos_r[:,:,2:3], sin_r[:,:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:,:,2:3], cos_r[:,:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 2)

    rot = rz.matmul(ry).matmul(rx)
    return rot

def _transform(x, theta): # by gzb: shape of x: (-1, self.seg, 75)
    # by gzb: input x, output x, and the shape is not changed
    # by gzb: x.size()[:2] + (-1, 3) means: new shape (sample_num, self.seg, -1, 3)
    x = x.contiguous().view(x.size()[:2] + (-1, 3))  # by gzb: shape of x: from (-1, self.seg, 75) to (-1, self.seg, 25, 3) (20210902)
    rot = x.new(x.size()[0], 3).uniform_(-theta, theta) # by gzb: shape of rot: (sample_num, 3), where "uniform_" means extract values from "Uniform distribution function"

    # by gzb: the following code line can be revided? to makethe  shape of rot is chaged from (sample_num, 3) into (sample_num, self.seg, 3)
    rot = rot.repeat(1, x.size()[1])  # by gzb: x.size()[1] is self.seg, so shape of rot is changed from (sample_num, 3) into (sample_num, 3 * self.seg)
    rot = rot.contiguous().view((-1, x.size()[1], 3)) # by gzb: shape is chaged from (sample, 3 * self.seg) into (-1, self.seg, 3)

    rot = _rot(rot)
    x = torch.transpose(x, 2, 3)
    x = torch.matmul(rot, x)
    x = torch.transpose(x, 2, 3)

    x = x.contiguous().view(x.size()[:2] + (-1,))  # by gzb: change the shape of x from (sample_num, self.seg, 25, 3) to (sample_num, self.seg, 75)
    return x
