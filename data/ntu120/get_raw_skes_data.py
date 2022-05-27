# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os.path as osp
import os
import numpy as np
import pickle
import logging


def get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger):
    """
    Get raw bodies data from a skeleton sequence.

    Each body's data is a dict that contains the following keys:
      - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
      - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
      - interval: a list which stores the frame indices of this body.
      - motion: motion amount (only for the sequence with 2 or more bodyIDs).

    Return:
      a dict for a skeleton sequence with 3 key-value pairs:
        - name: the skeleton filename.
        - data: a dict which stores raw data of each body.
        - num_frames: the number of valid frames.
    """
    # by gzb: from ctrgcn
    if int(ske_name[1:4]) >= 18:
        skes_path = '../nturgbd_raw/nturgb+d_skeletons120/'

    ske_file = osp.join(skes_path, ske_name + '.skeleton')  # by gzb: data file for skeleton information
    assert osp.exists(ske_file), 'Error: Skeleton file %s not found' % ske_file
    # Read all data from .skeleton file into a list (in string format)
    print('Reading data from %s' % ske_file[-29:])  # by gzb: filename, which just like "S032C003P067R002A075.skeleton", totally 29 char
    with open(ske_file, 'r') as fr:  # by gzb: read each skeleton file
        str_data = fr.readlines()

    num_frames = int(str_data[0].strip('\r\n'))  # by gzb: delete unuseful char, get frames number
    frames_drop = []
    bodies_data = dict()
    valid_frames = -1  # 0-based index
    current_line = 1

    for f in range(num_frames):
        num_bodies = int(str_data[current_line].strip('\r\n'))  # by gzb: because the second line( str_data[1] ) restores the number of actors
        current_line += 1

        if num_bodies == 0:  # no data in this frame, drop it
            frames_drop.append(f)  # 0-based index
            continue

        valid_frames += 1  # by gzb: valid_frames is initialed with -1, so it is begin with 0. which used to calculate the valid frames num
        joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)    # by gzb: for 3D
        colors = np.zeros((num_bodies, 25, 2), dtype=np.float32)    # by gzb: for 2D

        for b in range(num_bodies):  # by gzb: 1 or 2
            bodyID = str_data[current_line].strip('\r\n').split()[0] # by gzb: str_data[2] is the third line, which restore 10 different data, the 1st is bodyID
            current_line += 1  # by gzb: str_data[3] is the the forth line, which restore the num of joints
            num_joints = int(str_data[current_line].strip('\r\n'))  # 25 joints
            current_line += 1  # by gzb: str_data[4], the fifth line, which restore the first joint's info, 12 different info for each joint

            # by gzb: restore joint info and color info
            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32) # by gzb: x, y, z
                colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32) # by gzb: colorX, colorY
                current_line += 1  # by gzb: next joint

            # by gzb: restore info of the first valid_frame.
            if bodyID not in bodies_data:  # Add a new body's data
                body_data = dict()
                body_data['joints'] = joints[b]  # ndarray: (25, 3)
                body_data['colors'] = colors[b, np.newaxis]  # ndarray: (1, 25, 2)  # by gzb: add a new dimension; and why do this?
                body_data['interval'] = [valid_frames]  # the index of the first frame
            else:  # Update an already existed body's data  # by gzb: restore info based on "bodyID", actually, integrate info interms of "bodyID"
                body_data = bodies_data[bodyID]
                # Stack each body's data of each frame along the frame order
                body_data['joints'] = np.vstack((body_data['joints'], joints[b]))    # by gzb: why not indicate 'b'? reason: only restore joints[b] with ndarray: (25, 3)
                body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis]))
                pre_frame_idx = body_data['interval'][-1]  # by gzb: get index of previous frame
                body_data['interval'].append(pre_frame_idx + 1)  # add a new frame index

            # by gzb: bodies_data: {'bodyID': {'joints': XX-array, 'colors': xx-array, 'inteval': XX-int}}
            # by gzb: body_data: {'joints': XX-array, 'colors': xx-array, 'inteval': XX-int}
            bodies_data[bodyID] = body_data  # Update bodies_data

    # by gzb: prosessed droped frames and print info into log file
    num_frames_drop = len(frames_drop)
    # by gzb: for assert: if not satisfied this condition, print "error" info.
    assert num_frames_drop < num_frames, \
        'Error: All frames data (%d) of %s is missing or lost' % (num_frames, ske_name)  # by gzb: means this sample(video) is not useful.
    if num_frames_drop > 0:
        # by gzb: frames_drop_skes is a dict
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=np.int64)  # by gzb: a array with one dimension
        frames_drop_logger.info('{}: {} frames missed: {}\n'.format(ske_name, num_frames_drop,
                                                                    frames_drop))  # by gzb: frames_drop: index of droped frames

    # Calculate motion (only for the sequence with 2 or more bodyIDs)
    if len(bodies_data) > 1:  
        # by gzb: change body_data from {'joints': XX-array, 'colors': xx-array, 'inteval': XX-int} 
        # by gzb: to {'joints': [[]/n[]/n[]...], 'colors': [[]/n[]/n...], 'inteval': XX-int, 'motion': XX-int}
        # by gzb: np.var(body_data['joints'], axis=0) is used to calculate means of different row elements.
        for body_data in bodies_data.values():
            body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0)) # by gzb: for instance: shape from [25, 3] to 3 to 1

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}


def get_raw_skes_data():
    # # save_path = './data'
    # # skes_path = '/data/pengfei/NTU/nturgb+d_skeletons/'
    # stat_path = osp.join(save_path, 'statistics')
    #
    # skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    # save_data_pkl = osp.join(save_path, 'raw_skes_data.pkl')
    # frames_drop_pkl = osp.join(save_path, 'frames_drop_skes.pkl')
    #
    # frames_drop_logger = logging.getLogger('frames_drop')
    # frames_drop_logger.setLevel(logging.INFO)
    # frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'frames_drop.log')))
    # frames_drop_skes = dict()

    skes_name = np.loadtxt(skes_name_file, dtype=str)  # by gzb: read "statistics/skes_available_name_60/120.txt"

    num_files = skes_name.size # by gzb: is 56578 for ntu60; 56578+57065 for ntu120.

    print('Found %d available skeleton files.' % num_files)

    raw_skes_data = []
    frames_cnt = np.zeros(num_files, dtype=np.int64)  # by gzb: a one-dimension array, to restore frame num of each video samples.

    for (idx, ske_name) in enumerate(skes_name):
        # by gzb: read data from each skeleton file
        bodies_data2 = get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger)  # by gzb: is a dict: {'name': filename, 'data':XX, 'num_frames': XX}
        raw_skes_data.append(bodies_data2) # by gzb: a list
        frames_cnt[idx] = bodies_data2['num_frames']
        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d)' % \
                  (100.0 * (idx + 1) / num_files, idx + 1, num_files))

    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
    np.savetxt(osp.join(save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw bodies data into %s' % save_data_pkl)
    print('Total frames: %d' % np.sum(frames_cnt))

    # by gzb: store index info of each drop frame, which is similiar with frames_cnt
    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # by gzb:
    # input: ori data(such as S001C002P003R002A013.skeleton), skes_available_name.txt
    # output: restore into dir: "/raw_data/", including: raw_skes_data.pkl, frames_drop_skes.pkl, frames_cnt.txt (np.array), frames_drop.log


    save_path = './'

    #skes_path = './nturgb+d_skeletons/'
    #skes_path = '/home/gzb/gzb/datasets/actionRecognition/ntu17/'
    skes_path = '../nturgbd_raw/nturgb+d_skeletons/'
    stat_path = osp.join(save_path, 'statistics')
    if not osp.exists('./raw_data'):
        os.makedirs('./raw_data')

    skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    save_data_pkl = osp.join(save_path, 'raw_data', 'raw_skes_data.pkl')
    frames_drop_pkl = osp.join(save_path, 'raw_data', 'frames_drop_skes.pkl')

    frames_drop_logger = logging.getLogger('frames_drop')  # by gzb: get logger which named as 'frames_drop'
    frames_drop_logger.setLevel(logging.INFO)  # by gzb: set what info to be can be print. which can be debug, info, warning, error, criticial
    frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'raw_data', 'frames_drop.log'))) # by gzb: logging.FileHandler('aaa.log'), used to be written logs
    frames_drop_skes = dict()  # by gzb: eg: {"S001C002P003R002A013": [droped_frames_index1, droped_frames_index1, ...]}

    # by gzb
    #logging.StreamHandler()  # print log to console 

    get_raw_skes_data()

    # by gzb: write twice?
    #with open(frames_drop_pkl, 'wb') as fw:
    #    pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)

    # by gzb:
    # saved raw bodies data into ./raw_data/raw_skes_data.pkl
    # total frames: 4773093 for ntu60

    # by gzb:
    # for official ntu17: raw frames num is  4773093
    # for official ntu17: 113945 samples (useful), 4773093 frames
    # for official ntu01-32: 56578 samples (useful), 8140621
    # total samples: ntu60: 56880, missing 233; ntu120: 114488, missing 535

        
