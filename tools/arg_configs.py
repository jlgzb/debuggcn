import argparse

# by gzb: the following function is come from hcn or 2s-agcn
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_sgn_parser():
    parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
    # config
    parser.add_argument('--config', default='./configs/train_joint_gzb.yaml',
                        help='path to the configuration file')
    # by gzb: new added code (20210907)
    # data feeder
    parser.add_argument('--judge_rot', type=int, default=1,
                      help='the sub dir to restore train logs, record whether use rog in dataloader')
    parser.add_argument('--judge_dataloader', type=str, default='sgn',
                      help='judge the dataloader which is used in main.py. input shape with (N, self.seg, 150) for sgn; input shape with (N, C, T, V, M) for others.')

    parser.add_argument('--judge_GPU_set', type=int, default=0,
                      help='0 for local ubuntu system, 1 for gpu set')

    parser.add_argument('--h5_npz', type=int, default=1,
                      help='create data from h5 or npz file: 0 for h5, 1 for npz')


    parser.add_argument('--judge_dataSource', type=str, default='cs',
                      help='cs, cv for NTU60, csub, cset for ntu120')               
    parser.add_argument('--judge_inputDataShape', type=int, default=3,
                      help='3 for sgn with shape: (N, 300, 150); 5 for other with shape: (N, C, T, V, M)') 
                                                       
    parser.add_argument('--judge_dataType', type=int, default=0,
                      help='joint: 0; bone: 1')   # , joint_motion: 3; bone_motion: 4')
    parser.add_argument('--judge_dataMode', type=int, default=0,
                      help='0 for only joint|bone, 1 for t_joint|t_bone, 2 for joint + t_joint | bone + t_bone')

    parser.add_argument('--judge_binary', type=int, default=0,
                      help='0 for all sampels, 1 for partial sample')

    parser.add_argument('--train-feeder-args', default=dict(),
                        help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args', default=dict(),
                        help='the arguments of data loader for test')
    parser.add_argument('--train-feeder-paths', default=dict(),
                        help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-paths', default=dict(),
                        help='the arguments of data loader for test')

    parser.add_argument('--train_loader',
                        help='train data loader')
    parser.add_argument('--val_loader',
                        help='val data loader')                    
    parser.add_argument('--test_loader',
                        help='train data loader')    

    # processor
    parser.add_argument('--case', type=int, default=0,
                      help='select which case')
    parser.add_argument('--train', type=int, default=1,
                      help='train or test')
    parser.add_argument('--seg', type=int, default=20,
                      help='number of segmentation')
    parser.add_argument('--monitor', type=str, default='val_acc',
                      help='quantity to monitor (default: val_acc)')

    parser.add_argument('--debug', type=str, default=False,
                      help='select loade partial data, such as data[:1000] and label[:1000]')
    parser.add_argument('--mmap', type=str, default=False,
                      help='used for np.load')
    
    # model  
    parser.add_argument('--network', type=str, 
                      help='the neural network to use')          
    parser.add_argument('--network_label', type=int, default=0, 
                      help='the neural network record')

    parser.add_argument('--with_att', type=str, default=True, 
                      help='judge whether use selfattention')
    parser.add_argument('--with_block_tcn', type=str, default=True, 
                      help='judge whether use block_tcn which consist of multiscal cnn')
    parser.add_argument('--with_gcn_frm', type=str, default=True, 
                      help='judge whether use gcn for frame in block_gcn')
    parser.add_argument('--with_gcn_tcn_para', type=str, default=True, 
                      help='judge whether execute block_gcn and block_tcn paraently')

    # path
    parser.add_argument('--result_dir', type=str, 
                      help='dir to save chenckpoints')          

    # optim
    parser.add_argument('--dataset', type=str, default='NTU',
                      help='select dataset to evlulate, which can be ntu, ntu120, kinetics')
    parser.add_argument('--start_epoch', default=0, type=int,
                      help='manual epoch number (useful on restarts)')
    parser.add_argument('--max_epochs', type=int, default=120,
                      help='max number of epochs to run')
    parser.add_argument('--lr', type=float, default=0.1,
                      help='initial learning rate')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                      help='the ratio to reduce lr on each step')
    parser.add_argument('--weight_decay', '--wd', type=float, default=1e-4,
                      help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', '-p', type=int, default=10,
                      help='print frequency (default: 10)')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                      help='mini-batch size (default: 256)')             
    parser.add_argument('--num_classes', type=int, default=60,  # by gzb: 120 for ntu120, 60 for ntu60
                      help='the number of classes')
    parser.add_argument('--workers', type=int, default=2,
                      help='number of data loading workers (default: 2)')

    parser.add_argument('--resume', type=int, default=1,
                      help='1 for resume train')

    # train
    parser.add_argument('--milestones', type=int, nargs='+', default=[60, 90, 110], 
                      help='milestones for self.optimizer in training')
    parser.add_argument('--device', type=int, default=0,
                      help='indicate the num of GPU')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='mini-batch size (default: 256)')   
    parser.add_argument('--batch_size_test', type=int, default=32,
                      help='mini-batch size (default: 256)')
    parser.add_argument('--test_group', type=int, default=5,
                      help='test data: 32*5')
    parser.add_argument('--split_csub', type=str, default=False,
                      help='split xsub of ntu120, get data of labels 60-119')
    
    return parser


