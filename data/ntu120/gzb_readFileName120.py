import os
import numpy as np
import os.path as osp
import h5py
# get file name of ntu rgb-d 120, return a list
def getFileNameList120(fileName):
    list_filePath = os.listdir(fileName)
    print (len(list_filePath))

    list_names = []
    # get file names
    for _path in list_filePath:
        availableName = _path.split('/')[-1].strip()
        availableName = availableName.split('.')[0].strip()
        list_names.append(availableName)

        '''
        i += 1
        if i == 10:
            print (list_name)

            list_name.sort()
            print (list_name)
            break
        '''

    list_names.sort()

    return list_names

def wright2txt(list_names, restore_path):
    # wright file name into txt file
    #restore_path = '/home/gzb/gzb/actionRecModels/sgn/data/ntu/120.txt'
    fout = open(restore_path, 'w')
    for str_fileName in list_names:
        fout.write(str_fileName + '\n')
    fout.close()

    # np.savetxt($path, $list, fmt="%d", delemiter="\n" )

def readTxt(fin_path):
    '''
    fin = open(fin_path, 'r')
    #missingName = np.loadtxt('fin_path')
    list_missingNames = fin.readlines()
    
    # delete '\n' for each elem
    list_newMissingNames = [elem.strip() for elem in list_missingNames]
    list_newMissingNames.sort()

    #print (list_newMissingNames[:10])
    return list_newMissingNames
    '''
    
    list_missingNames = np.loadtxt(fin_path, dtype=str)
    list_missingNames.sort()
    return list_missingNames
    
def deleteMissingSamples(list_names, list_missingNames):
    list_newNames = []
    list_newNames = [elem for elem in list_names if elem not in list_missingNames]

    #for _elem in list_names:
    #    if _elem not in list_missingNames:
    #        list_newNames.append(_elem)

    list_newNames.sort()

    return list_newNames


def getSetupInfo(finPath, foutPath):
    skes_name = np.loadtxt(finPath, dtype=str) # a list
    num_files = skes_name.size # by gzb: is 56578 for ntu60; 56578+57065 for ntu120.
    print('Found %d available skeleton files.' % num_files)

    list_setup = []

    for ske_name in skes_name:
        setupID = np.int64(ske_name[1:4]) # for "S032C003P106R002A120", extract "032" of S
        list_setup.append(setupID)

    np.savetxt(foutPath, list_setup, fmt="%d")

def getCameraInfo(finPath, foutPath):
    skes_name = np.loadtxt(finPath, dtype=str) # a list
    num_files = skes_name.size # by gzb: is 56578 for ntu60; 56578+57065 for ntu120.
    print('Found %d available skeleton files.' % num_files)

    list_camera = []
    for ske_name in skes_name:
        cameraID = np.int64(ske_name[5:8]) # for "S032C003P106R002A120", extract "003" for C
        list_camera.append(cameraID)

    np.savetxt(foutPath, list_camera, fmt="%d")

def getPerformerInfo(finPath, foutPath):
    skes_name = np.loadtxt(finPath, dtype=str) # a list
    num_files = skes_name.size # by gzb: is 56578 for ntu60; 56578+57065 for ntu120.
    print('Found %d available skeleton files.' % num_files)

    list_performer = []
    for ske_name in skes_name:
        performerID = np.int64(ske_name[9:12]) # for "S032C003P106R002A120", extract "106" for P
        list_performer.append(performerID)

    np.savetxt(foutPath, list_performer, fmt="%d")

def getReplicationInfo(finPath, foutPath):
    skes_name = np.loadtxt(finPath, dtype=str) # a list
    num_files = skes_name.size # by gzb: is 56578 for ntu60; 56578+57065 for ntu120.
    print('Found %d available skeleton files.' % num_files)

    list_replication = []
    for ske_name in skes_name:
        republicationID = np.int64(ske_name[13:16]) # for "S032C003P106R002A120", extract "002" for R
        list_replication.append(republicationID)

    np.savetxt(foutPath, list_replication, fmt="%d")

def getLabelInfor(finPath, foutPath):
    skes_name = np.loadtxt(finPath, dtype=str) # a list
    num_files = skes_name.size # by gzb: is 56578 for ntu60; 56578+57065 for ntu120.
    print('Found %d available skeleton files.' % num_files)

    list_label = []
    for ske_name in skes_name:
        labelID = np.int64(ske_name[-3:]) # for "S032C003P106R002A120", extract "120" for A
        list_label.append(labelID)

    np.savetxt(foutPath, list_label, fmt="%d")

def showMaxMinNum(frame_cnt_path):
    list_frameCnt = np.loadtxt(frame_cnt_path, dtype=np.int64)
    maxNum = list_frameCnt.max()
    minNum = list_frameCnt.min()
    #print (list_frameCnt[:10])

    print ("unique len: {}".format(len(np.unique(list_frameCnt))))
    print ("unique: {}".format(np.unique(list_frameCnt)))

    exit()

    print ("max is {}.".format(maxNum))
    print ("min is {}.".format(minNum))

    i = 0
    for idx, l in enumerate(list_frameCnt):
        print (idx, l)
        i+= 1

        if i == 10:
            break

    # for test which can be delete
    #print (np.unique(list_frameCnt))
    #print (np.where(list_frameCnt == 6)[0]) # print index

def readFromH5(h5_path):
    '''
    datataset shape      train_X    | train_Y  |     val_X      |  val_Y  |     test_X      | testY    |
    ntu60       CS  (38086,300,150) | (38086,) | (2005,300,150) | (2005,) | (16487.300,150) | (16487,) |  56578 samples
                CV  (35763,300,150) | (35763,) | (1883,300,150) | (1883,) | (18932,300,150) | (18932,) |
    ntu120      CS  (59874,300,150) | (59874,) | (3152,300,150) | (3251,) | (50919,300,150) | (50519,) |  113945 samples
                CV  (72031,300,150) | (72031,) | (3792,300,150) | (3792,) | (38122,300,150) | (38122,) |  113945
    oneActor    CS  (46958,300,75)  | (46958,) | (2472,300,75)  | (2472,) | (38203,300,75)  | (38203,) |  87633 samples
                CV  (55402,300,75)  | (55402,) | (2916,300,75)  | (2916,) | (29315,300,75)  | (29315,) |
    '''
    
    print ("GZB: Load data from h5 file in gzb_readFileName120.py...")  # by gzb: now added code.
    f = h5py.File(h5_path , 'r')
    train_X = f['x'][:]   # by gzb: shape of train_x: (train_sample_num, 300, 150)
    # by gzb: translate one-hot vector into int num; Concretely, for label shape: from (train_sample_num, 120) to (train_sample_num,)
    train_Y = np.argmax(f['y'][:],-1) 
    val_X = f['valid_x'][:]
    val_Y = np.argmax(f['valid_y'][:], -1)
    test_X = f['test_x'][:]
    test_Y = np.argmax(f['test_y'][:], -1)
    f.close()
    print ("GZB: End load data from h5 file in gzb_readFileName120.py !!")  # by gzb: now added code.

    print ("GZB: the type of dataset from h5 is: {}, {} !".format(type(train_X), type(train_Y)))

    print ("GZB: the shape of train_X, train_Y, val_X, val_Y, test_X, test_Y is {}, {}, {}, {}, {}, {} !".format(
         train_X.shape, train_Y.shape, val_X.shape, val_Y.shape, test_X.shape, test_Y.shape))

    print ("GZB: the label class of train_Y, val_Y, test_Y are {}\n, {}\n, and {}\n !".format(np.unique(train_Y), np.unique(val_Y), np.unique(test_Y)))

if __name__ == "__main__":
    save_path = './'
    stat_path = osp.join(save_path, 'statistics')  # "./statistics/"
    denoised_path = osp.join(save_path, 'denoised_data')

    fileName_path = '/home/gzb/gzb/datasets/actionRecognition/ntu17/'
    missingName_path = '/home/gzb/gzb/actionRecModels/sgn/data/ntu/statistics/samples_with_missing_skeletons_120gzb.txt'

    # by gzb: then, mv 120_withoutMissing.txt skes_available_name_120.txt
    restore_path = '/home/gzb/gzb/actionRecModels/sgn/data/ntu/statistics/120_withoutMissing.txt'

    skes_name_file = osp.join(stat_path, 'skes_available_name_120.txt')
    setup_path = osp.join(stat_path, 'setup_120.txt')
    camera_path = osp.join(stat_path, 'camera_120.txt')
    performer_path = osp.join(stat_path, 'performer_120.txt')
    replication_path = osp.join(stat_path, 'replication_120.txt')
    label_path = osp.join(stat_path, 'label_120.txt')

    # 20210904: test the ID of performer and replication
    _path = osp.join(stat_path, 'performer.txt')
    showMaxMinNum(_path)

    # 20210903: test the shape of dataset from h5py
    metric = "CV"
    h5_path = osp.join('../ntu120/h5_oneActor', 'NTU_' + metric + '.h5')
    #readFromH5(h5_path)

    # 2021.09.01 for setup_120.txt, label_120.txt, performer_120.txt, camera_120.txt, and replication.txt
    #getSetupInfo(skes_name_file, setup_path)
    #getCameraInfo(skes_name_file, camera_path)
    #getPerformerInfo(skes_name_file, performer_path)
    #getReplicationInfo(skes_name_file, replication_path)
    #getLabelInfor(skes_name_file, label_path)

    '''
    # 2021.09.01, show the max frames num and min frames num of all samples
    frameCnt_path = osp.join(denoised_path, 'frames_cnt.txt')
    frameCnt60_path = osp.join('/home/gzb/gzb/actionRecModels/sgn/data/ntu/denoised_data', 'frames_cnt.txt')
    showMaxMinNum(label_path)
    '''

    ''' # 2021.08.31
    # to generate skes_available_name_120.txt
    # get total file names: 114480
    list_names = getFileNameList120(fileName_path)
    totalNum = len(list_names)
    print ("Total names num is {} !".format(totalNum))

    # get missing file names: 535
    list_missingNames = readTxt(missingName_path)
    missingNum = len(list_missingNames)
    print ("Missing names num is {} !".format(missingNum))

    # get useful names: total names minus missing names, which is 113945
    list_newNames = deleteMissingSamples(list_names, list_missingNames)
    newNum = len(list_newNames)
    print ("Usefule names num is {} !".format(newNum))

    # wright to tx
    wright2txt(list_newNames, restore_path)
    '''

    ''' # 2021.08.31
    # test np.loadtxt and enumerate func
    skes_name = np.loadtxt(missingName_path, dtype=str)
    #print (len(skes_name))
    #print (skes_name[:10])
    for (idx, ske_name) in enumerate(skes_name):
        print (idx, ske_name)
        break
    '''











