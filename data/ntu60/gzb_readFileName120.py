import os
import numpy as np
import os.path as osp

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
    list_frameCnt = np.loadtxt(frame_cnt_path)
    maxNum = list_frameCnt.max
    minNum = list_frameCnt.min

    print ("max is {}.".format(maxNum))
    print ("min is {}.".format(minNum))

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

    # 2021.09.01 for setup_120.txt, label_120.txt, performer_120.txt, camera_120.txt, and replication.txt
    #getSetupInfo(skes_name_file, setup_path)
    #getCameraInfo(skes_name_file, camera_path)
    #getPerformerInfo(skes_name_file, performer_path)
    #getReplicationInfo(skes_name_file, replication_path)
    #getLabelInfor(skes_name_file, label_path)

    # 2021.09.01, show the max frames num and min frames num of all samples
    frameCnt_path = osp.join(denoised_path, 'frames_cnt.txt')
    showMaxMinNum(frameCnt_path)



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











