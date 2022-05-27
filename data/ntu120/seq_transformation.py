# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle
import logging
import h5py
from sklearn.model_selection import train_test_split

root_path = './'
stat_path = osp.join(root_path, 'statistics')
setup_file = osp.join(stat_path, 'setup.txt')
camera_file = osp.join(stat_path, 'camera.txt')
performer_file = osp.join(stat_path, 'performer.txt')
replication_file = osp.join(stat_path, 'replication.txt')
label_file = osp.join(stat_path, 'label.txt')
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')

denoised_path = osp.join(root_path, 'denoised_data')
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(denoised_path, 'frames_cnt.txt')

save_path = './'

# by gzb: 20211229 for get csub data of labels from 60-119
split_csub = False

if not osp.exists(save_path):
    os.mkdir(save_path)

# by gzb: new added code
def createDir(dirPath):
    if not osp.exists(dirPath):
        os.mkdir(dirPath)


def remove_nan_frames(ske_name, ske_joints, nan_logger):
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))

    return ske_joints[valid_frames]

def seq_translation(skes_joints):
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2  # by gzb: the dim of vector, 75 dim for one actor; 150 for two actor

        # by gzb: process joint info of actor2. step 1: record the index of frames of which the actor's joint info are all zero
        if num_bodies == 2:
            # by gzb: a list which restore index of frame
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]  # by gzb: which have been done in script "get_row_denoised_data.py"
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1) # by gzb: frame num of which actor1's info is useless
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):  # by gzb: the joint position of actor is not equal to '0', which means that this sample is useful (can be used)
                break
            i += 1

        # by gzb: the second keypoint (middle of the spine). Here 'i' is the index of 'real' first frame ; 0-based
        origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

        # by gzb: Take "middle of spine" joint as the origin of corrdinate
        # by gzb: step 2: calculate the position (without second joint) of the relative coordinate origin (second joint)
        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)  # by gzb: expand origin 25 times, get shape: (75,)
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)  # by gzb: expand origin 50 times, get shape: (150, )

        # by gzb: step 3: now the joint info of missing frames is negative value, which need to be reset to zero
        if (num_bodies == 2) and (cnt1 > 0):  # by gzb: process sample which have two actor in its frames
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)  # by gzb: initial with '0' array? Yes !(20210902)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

        skes_joints[idx] = ske_joints  # Update  # by gzb: update sample with index 'idx'

    return skes_joints  # by gzb: skes_joints(sample_num, frames_num, 75 or 150): the joints info of each frame is "middle of spine"-joint-based

# by gzb: new add function, which used to extract samplesID which performed by one actor
def getOneActorSamples(skes_joints):
    list_sampleID_withOneActor = []
    for idx, ske_joints in enumerate(skes_joints):
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2  # by gzb: the dim of vector, 75 dim for one actor; 150 for two actor
        if num_bodies == 1: # only one actor
            list_sampleID_withOneActor.append(idx)

    return list_sampleID_withOneActor  # index of samples which is single action (one actor).

# by gzb: ori code, which are not used in this script?
def frame_translation(skes_joints, skes_name, frames_cnt):
    nan_logger = logging.getLogger('nan_skes')
    nan_logger.setLevel(logging.INFO)
    nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
    nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        # Calculate the distance between spine base (joint-1) and spine (joint-21)
        j1 = ske_joints[:, 0:3]
        j21 = ske_joints[:, 60:63]
        dist = np.sqrt(((j1 - j21) ** 2).sum(axis=1))

        for f in range(num_frames):
            origin = ske_joints[f, 3:6]  # new origin: middle of the spine (joint-2)
            if (ske_joints[f, 75:] == 0).all():
                ske_joints[f, :75] = (ske_joints[f, :75] - np.tile(origin, 25)) / \
                                      dist[f] + np.tile(origin, 25)
            else:
                ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)) / \
                                 dist[f] + np.tile(origin, 50)

        ske_name = skes_name[idx]
        ske_joints = remove_nan_frames(ske_name, ske_joints, nan_logger)
        frames_cnt[idx] = num_frames  # update valid number of frames
        skes_joints[idx] = ske_joints

    return skes_joints, frames_cnt

# by gzb: the joint info is changed after the following "align_frame" func
# by gzb: for single actor, shape of each sample(video) is changed from (num_frames, 75) to (num_frames, 150) by add neros
# by gzb: for interactive action, shape of one sample (video)  is not changed, s.t. (num_frames, 150)
def align_frames(skes_joints, frames_cnt):
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)  # by gzb: the num of samples? yes!!(20210901) 113945 for ntu120 and 56578 for ntu60
    
    # by gzb: max is 300 and min is 10 for ntu120
    max_num_frames = frames_cnt.max()  # 300  
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32) # by gzb: the vector of all samples (videos)?  Yes !!! (20210901)

    # by gzb: shape of skes_joints: (sampleID, frame_num, 75 or 150)
    for idx, ske_joints in enumerate(skes_joints):  # by gzb: ske_joints is the joint info of one frame
        num_frames = ske_joints.shape[0]  # by gzb: frames num
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            # by gzb: add vector of missing actor with zero
            # by gzb: for ske_joints with shape (num_frames, 75), expand ske_joints 2 times (num_frames, 150) and align to aligned_skes_joints.
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints,
                                                               np.zeros_like(ske_joints)))  # by gzb: ske_joins, np.zeros_like(ske_joints)
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints  # by gzb: info 

    # by gzb: has zeros if frames of each sample < max_num_frames
    return aligned_skes_joints  # by gzb: shape is (samples_num, max_num_frames, 150)

# by gzb: new added func by gzb
def align_frames_for_single_actor(skes_joints, frames_cnt):
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)  # by gzb: sample num
    max_num_frames = frames_cnt.max()  # 300  ?
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 75), dtype=np.float32)
    for idx, ske_joints in enumerate(skes_joints):  # by gzb: ske_joints is the joint info of one frame
        num_frames = ske_joints.shape[0]
        aligned_skes_joints[idx, :num_frames] = ske_joints  # by gzb: since only one actor, do not need add zeros throuth np.hstack((ske_joints, np.zeros_like(ske_joints)))

    return aligned_skes_joints  # by gzb: which complete added frames of which the joint info are all zeros

# by gzb: int values in labels have been minus 1
def one_hot_vector(labels):
    num_skes = len(labels)

    # by gzb: 60 for ntu60, ori code
    #labels_vector = np.zeros((num_skes, 60))  # by gzb: the vector of each vector is 60 since there are 60 classes
    # by gzb: 120 for ntu120, ori code

    if split_csub == False:
        labels_vector = np.zeros((num_skes, 120))  # by gzb: shape of label change from (num_skes,) to (num_skes, 120)

        for idx, l in enumerate(labels):  # by gzb: "l" is the int value, such as 59 for ntu60, 119 for ntu120, which means the actionID (class number).
            labels_vector[idx, l] = 1
    
    else:
        labels_vector = np.zeros((num_skes, 60))
        for idx, l in enumerate(labels): # l: from 60 to 119
            labels_vector[idx, l - 60] = 1

    return labels_vector


def split_train_val(train_indices, method='sklearn', ratio=0.05):
    """
    Get validation set by splitting data randomly from training set with two methods.
    In fact, I thought these two methods are equal as they got the same performance.

    """
    if method == 'sklearn':
        return train_test_split(train_indices, test_size=ratio, random_state=10000)
    else:
        np.random.seed(10000)
        np.random.shuffle(train_indices)
        val_num_skes = int(np.ceil(0.05 * len(train_indices)))
        val_indices = train_indices[:val_num_skes]
        train_indices = train_indices[val_num_skes:]
        return train_indices, val_indices


def split_dataset(skes_joints, label, performer, camera, evaluation, save_path):
    train_indices, test_indices = get_indices(performer, camera, evaluation)  # by gzb: get index of samples and retore into np.ndarray

    #'''
    # by gzb: ori code from CTR-GCN 20211202
    # Save labels and num_frames for each sequence of each data set
    train_labels = label[train_indices]
    test_labels = label[test_indices]

    train_x = skes_joints[train_indices]
    test_x = skes_joints[test_indices]

    # by gzb: get data with labels from 60-119
    if evaluation == 'CSub' and split_csub:
        # for train data
        idx_train = np.where(train_labels > 59)
        train_x = train_x[idx_train]
        train_labels = train_labels[idx_train]

        # for test data
        idx_test = np.where(test_labels > 59)
        test_x = test_x[idx_test]
        test_labels = test_labels[idx_test]
    
    train_y = one_hot_vector(train_labels)
    test_y = one_hot_vector(test_labels)

    save_name = 'NTU_%s.npz' % evaluation
    np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)
    #'''

    ''' ori code from sgn
    m = 'sklearn'  # 'sklearn' or 'numpy'
    # Select validation set from training set
    train_indices, val_indices = split_train_val(train_indices, m)  # by gzb: train:test == 0.95:0.05(ratio)  # update train_indices

    # Save labels and num_frames for each sequence of each data set
    train_labels = label[train_indices]
    val_labels = label[val_indices]
    test_labels = label[test_indices]

    # Save data into a .h5 file
    h5file = h5py.File(osp.join(save_path, 'NTU_%s.h5' % (evaluation)), 'w')  # by gzb: create h5 file
    # Training set
    h5file.create_dataset('x', data=skes_joints[train_indices])  # by gzb: add train data into h5 file
    train_one_hot_labels = one_hot_vector(train_labels)  # by gzb: translate label into one-hot vector
    h5file.create_dataset('y', data=train_one_hot_labels)  # by gzb: add train label into h5 file
    # Validation set
    h5file.create_dataset('valid_x', data=skes_joints[val_indices])
    val_one_hot_labels = one_hot_vector(val_labels)
    h5file.create_dataset('valid_y', data=val_one_hot_labels)
    # Test set
    h5file.create_dataset('test_x', data=skes_joints[test_indices])
    test_one_hot_labels = one_hot_vector(test_labels)
    h5file.create_dataset('test_y', data=test_one_hot_labels)

    h5file.close()
    '''

# by gzb: used to get index of train samples and test samples; type is numpy.ndarray
def get_indices(performer, setup, evaluation='CSub'):
    test_indices = np.empty(0)  # by gzb: create array: array([], dtype=float64), shape is  (0,)
    train_indices = np.empty(0)

    '''
    # by gzb: total 40 subjects for ntu60, ori code
    if evaluation == 'CS':  # Cross Subject (Subject IDs)
        train_ids = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
                     17, 18, 19, 25, 27, 28, 31, 34, 35, 38] # by gzb: 20 subjects
        test_ids = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
                    24, 26, 29, 30, 32, 33, 36, 37, 39, 40]  # by gzb: 20 subjects
    '''
    
    # by gzb: total 106 subjects for ntu120, add by gzb
    if evaluation == 'CSub':  # Cross Subject (Subject IDs)
        train_ids = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
                     17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
                     45, 46, 47, 49, 50, 52, 53, 54, 55, 56,
                     57, 58, 59, 70, 74, 78, 80, 81, 82, 83,
                     84, 85, 86, 89, 91, 92, 93, 94, 95, 97,
                     98, 100, 103]  # by gzb: 53 subjects
        test_ids = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
                    24, 26, 29, 30, 32, 33, 36, 37, 39, 40,
                    41, 42, 43, 44, 48, 51, 60, 61, 62, 63,
                    64, 65, 66, 67, 68, 69, 71, 72, 73, 75,
                    76, 77, 79, 87, 88, 90, 96, 99, 101, 102,
                    104, 105, 106]  # by gzb: 53 subjects

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index  # by gzb: performer is a list. Here 'temp' is a list which restore index of "performer==0"
            test_indices = np.hstack((test_indices, temp)).astype(np.int64)  # by gzb: restore the index of test performerID (s.t. subjectID)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(np.int64)
    else:
        '''# Cross View (Camera IDs)
        train_ids = [2, 3]
        test_ids = 1
        # Get indices of test data
        temp = np.where(camera == test_ids)[0]  # 0-based index
        test_indices = np.hstack((test_indices, temp)).astype(np.int64)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(camera == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(np.int64)
        '''

        # Cross Setup for ntu120
        train_ids = [i for i in range(1, 33) if i % 2 == 0]  # Even setup
        test_ids = [i for i in range(1, 33) if i % 2 == 1]  # Odd setup

        # Get indices of test data
        for test_id in test_ids:
            temp = np.where(setup == test_id)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(np.int64)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(setup == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(np.int64)


    return train_indices, test_indices


if __name__ == '__main__':
    ''' by gzb: 2021-12-15
    input: camera_file, performer_file, label_file, frames_file, skes_name_file, raw_skes_joints_pkl
    output: NTU_CSub.npz, NTU_CSet.npz 20211215;    ori: NTU_CSub.h5, NTU_CSet.h5
        CSub: cross Subject (Subject IDs == performer); 
        CSet: cross setup, for ntu60. 20211203
        CV: cross view (Camera IDs), 2, 3 for train; 1 for test
    '''

    #camera = np.loadtxt(camera_file, dtype=np.int64)  # camera id: 1, 2, 3, type is np.ndarray
    setup = np.loadtxt(setup_file, dtype=np.int64)  # camera id: 1~32
    performer = np.loadtxt(performer_file, dtype=np.int64)  # subject id: 1~40
    label = np.loadtxt(label_file, dtype=np.int64) - 1  # action label: 0~59 # by gzb: all elements subtract 1  # by gzb: 0~119 for ntu120

    frames_cnt = np.loadtxt(frames_file, dtype=np.int64)  # frames_cnt    # by gzb: valid frames num for different samples
    skes_name = np.loadtxt(skes_name_file, dtype=np.string_)  # by gzb: fileName, such as "S032C003P106R002A120"

    # by gzb: joints shape in skes_joints: (num_frames, 75), or (num_frames1, 150); num_frames1 means remove missing frames
    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list

    '''
    # by gzb: test whether the 1rd demension of skes_joints is samples_num or performer_num(subject), conclusion: YES !
    skes_joints_array = np.array(skes_joints)
    print (skes_joints_array.shape)
    print (skes_joints_array.shape[0])
    '''

    #''' # by gzb: ori code
    skes_joints = seq_translation(skes_joints)  # by gzb: process joint info: Take "middle of spine" joint as the origin of corrdinate

    skes_joints = align_frames(skes_joints, frames_cnt)  # aligned to the same frame length  # by gzb: shape is (113945, 300, 150), type: np.ndarray

    if not split_csub:
        evaluations = ['CSub', 'CSet']
    else:
        evaluations = ['CSub']

    for evaluation in evaluations:
        split_dataset(skes_joints, label, performer, setup, evaluation, save_path)
    #'''

    ''' by gzb: not complete, need TODO: edit "split_dataset" func
    ## by gzb: TODO: extract the samples which performed by single actor
    list_oneActorSampleID = getOneActorSamples(skes_joints)
    skes_joints = np.array(skes_joints)[list_oneActorSampleID]  # shape: (-1, -1, 75). tranfom list into list
    frames_cnt_oneActor = frames_cnt[list_oneActorSampleID]
    skes_joints = align_frames_for_single_actor(skes_joints, frames_cnt_oneActor)  # shape: (87633, 300, 75)
    print ("Shape of single action sample: {} !".format(skes_joints.shape))
    
    # by gzb: split_dataset
    label = label[list_oneActorSampleID]
    print ("GZB: the label class is: {} !".format(np.unique(label)))
    performer = performer[list_oneActorSampleID]
    camera = camera[list_oneActorSampleID]
    print ("lenth of label, performer, and camera are {}, {}, and {} !".format(len(label), len(performer), len(camera)))
    
    save_path = './h5_oneActor/'
    createDir(save_path)
    
    evaluations = ['CS', 'CV']
    for evaluation in evaluations:  # by gzb: need to change save path name: e.g. "NTU_CS_actor1.h5"
        split_dataset(skes_joints, label, performer, camera, evaluation, save_path)

    '''


