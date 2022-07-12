
from torch import nn
import torch
import math
from tools.utils_main import save_arg

#from tools.utils_attention import SpatialAttention, SelfAtt_Spatial
#from tools.utils_attention import PyramidSplitAtt, BAMBlock, DuableAtt, CoordAtt
#from tools.utils_attention import DualAtt, SelfAtt_channel, ShuffleAtt, VisionPermuteMLP, PolarizedSelfAtt
from tools.utils_selfAtt import SelfAtt
from models.tools_cigcn.tool_inputs import block_idx_info, block_joint_input, block_bone_input
from models.tools_cigcn.tool_graph import unit_adjacent, conv_init, bn_init
from models.tools_cigcn.tool_tcn_gcn import unit_tcn, MultiScale_tcn, module_gcn, MultiScale_gcn

from models.graph.ntu_rgb_d import Graph

class module_adjacent(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.adj_joint = unit_adjacent(in_channels, out_channels, mode='joint')
        self.adj_frame = unit_adjacent(in_channels, out_channels, mode='frame')

    def forward(self, input):
        '''
        shape:
            input: (N, in_C, V, T)
            output (mode='joint'): (N, in_C, V, V) and (N, out_C, V, V)
            output (mode='frame'): (N, in_C, T, T) and (N, out_C, T, T)
        '''
        adaA_joint = self.adj_joint(input) # (N, out_C, 25, 25)
        adaA_frame = self.adj_frame(input) # (N, out_C, 20, 20)

        return adaA_joint, adaA_frame

class module_selfAtt(nn.Module):
    def __init__(self, in_channels, out_channels, mode='spatial'):
        super(module_selfAtt, self).__init__()

        self.conv_att_1 = SelfAtt(in_channels, out_channels, mode=mode)
        self.conv_att_2 = SelfAtt(out_channels, out_channels, mode=mode)
    
    def forward(self, input):
        '''
        shape:
            input: (N, in_channels, V, T)
            output: (N, out_channel, V, T)
        '''
        out = self.conv_att_1(input)
        out = self.conv_att_2(out)
        return out

class block_gcn(nn.Module):
    def __init__(self, in_channels=128, out_channels=256 , add_idxFrm=True, A=None, attention=True, gcn_frm=True):
        super().__init__()
        self.add_idxfrm = add_idxFrm

        # 20211229
        self.attention = attention
        self.gcn_frm = gcn_frm

        ##### 1 for GCN module
        # 1.1 module adjacent
        self.module_adaA = module_adjacent(in_channels, in_channels)
        #self.module_adaA_bone = module_adjacent(128, 128)
                
        # 1.2 module gcn
        self.module_gcn_joint = module_gcn(in_channels, out_channels, mode='joint')
        self.module_gcn_frame = module_gcn(in_channels, out_channels, mode='frame') if self.gcn_frm else 0
        #self.module_gcn_joint = MultiScale_gcn(128, 256, A)

        # 1.3 module attention
        self.module_att_joint = module_selfAtt(out_channels, out_channels, mode='spatial') if self.attention else 0
        self.module_att_frame = module_selfAtt(out_channels, out_channels, mode='spatial') if self.gcn_frm and self.attention else 0


    def forward(self, input, idx_frm):
        ##### 1 for GCN module
        # 1.1 ada_A
        adaA_joint, adaA_frame = self.module_adaA(input) # (N, -1, 25, 25) and (N, -1, 20, 20)
 
        # 1.2 gcn
        gcn_joint = self.module_gcn_joint(input, adaA_joint) # (N, 128, V, T)
        gcn_frame = self.module_gcn_frame(input, adaA_frame) if self.gcn_frm else 0 # (N, 256, V, T)        

        # 1.3 attention
        gcn_joint = self.module_att_joint(gcn_joint) if self.attention else 0  # (N, 256, V, T)
        gcn_frame = self.module_att_frame(gcn_frame) if self.gcn_frm and self.attention else 0 # (N, 256, V, T)
        #gcn_input = torch.cat([gcn_joint, gcn_frame], dim=1) # (N, 256, V, T)

        if self.add_idxfrm:
            gcn_output = gcn_joint + gcn_frame + idx_frm
        else:
            gcn_output = gcn_joint + gcn_frame

        return gcn_output #, adaA_joint, adaA_frame

class block_gcn_debug(nn.Module):
    def __init__(self, in_channels=128, out_channels=256 , add_idxFrm=True, A=None, attention=True, gcn_frm=True):
        super().__init__()
        self.add_idxfrm = add_idxFrm

        # 20211229
        self.attention = attention
        self.gcn_frm = gcn_frm

        ##### 1 for GCN module
        # 1.1 module adjacent
        self.module_adaA = module_adjacent(in_channels, in_channels)


        if gcn_frm:
            # 1.2 module gcn
            self.module_gcn_joint = module_gcn(in_channels, out_channels, mode='joint')
            self.module_gcn_frame = module_gcn(in_channels, out_channels, mode='frame')

            # 1.3 module attention
            if attention:
                self.module_att_joint = module_selfAtt(out_channels, out_channels, mode='spatial')
                self.module_att_frame = module_selfAtt(out_channels, out_channels, mode='spatial')

        if not gcn_frm:
            # 1.2 module gcn
            self.module_gcn_joint = module_gcn(in_channels, out_channels, mode='joint')

            # 1.3 module attention
            if attention:
                self.module_att_joint = module_selfAtt(out_channels, out_channels, mode='spatial')


    def forward(self, input, idx_frm):
        ##### 1 for GCN module
        # 1.1 ada_A
        adaA_joint, adaA_frame = self.module_adaA(input) # (N, -1, 25, 25) and (N, -1, 20, 20)

        if self.gcn_frm:
            # 1.2 gcn
            gcn_joint = self.module_gcn_joint(input, adaA_joint) # (N, 128, V, T)
            gcn_frame = self.module_gcn_frame(input, adaA_frame) # (N, 256, V, T)

            # 1.3 attention
            if self.attention:
                gcn_joint = self.module_att_joint(gcn_joint) # (N, 256, V, T)
                gcn_frame = self.module_att_frame(gcn_frame) # (N, 256, V, T)

            if self.add_idxfrm:
                gcn_output = gcn_joint + gcn_frame + idx_frm
            else:
                gcn_output = gcn_joint + gcn_frame

        if not self.gcn_frm:
            # 1.2 gcn
            gcn_joint = self.module_gcn_joint(input, adaA_joint) # (N, 128, V, T)

            # 1.3 attention
            if self.attention:
                gcn_joint = self.module_att_joint(gcn_joint) # (N, 256, V, T)

            if self.add_idxfrm:
                gcn_output = gcn_joint + idx_frm
            else:
                gcn_output = gcn_joint

        return gcn_output #, adaA_joint, adaA_frame


class block_tcn(nn.Module):
    def __init__(self, in_channels=128,  out_channels=256, add_idxFrm=True):
        super().__init__()
        self.add_idxfrm = add_idxFrm

        #self.module_adaA = module_adjacent(128, 128)

        self.multiscale_tcn_joint = MultiScale_tcn(in_channels, out_channels, kernel_size=[1, 3, 3, 5], dilations=[1, 1, 2, 2], mode='joint')
        self.multiscale_tcn_frame = MultiScale_tcn(in_channels, out_channels, kernel_size=[1, 3, 3, 5], dilations=[1, 1, 2, 2], mode='frame')

    def forward(self, input, idx_frm):
        ##### 2 for TCN module
        '''
        # 2.1 ada_A_T
        adaA_joint_T, adaA_frame_T = self.module_adaA_tcn(input_T) # (N, seg, 25, 25) and (N, seg, 128, 128)

        # 2.2 multiscale tcn
        tcn_joint = self.multiscale_tcn_joint_T(input_T, adaA_joint_T) # (N, T, V, 128)
        tcn_frame = self.multiscale_tcn_frame_T(input_T, adaA_frame_T) # (N, T, V, 128)

        # 2.3 attention T
        tcn_joint = self.module_att_joint_T(tcn_joint).permute(0, 3, 2, 1).contiguous() # (N, 128, V, T)
        tcn_frame = self.module_att_frame_T(tcn_frame).permute(0, 3, 2, 1).contiguous() # (N, 128, V, T)
        
        # 2.4 tcn output
        tcn_output = torch.cat([tcn_joint, tcn_frame], dim=1) # (N, 256, V, T)
        '''

        # 1.1 ada_A
        #adaA_joint, adaA_frame = self.module_adaA(input) # (N, -1, 25, 25) and (N, -1, 20, 20)

        # 2.2 multiscale tcn
        tcn_joint = self.multiscale_tcn_joint(input, 0) # (N, 256, V, T)
        #tcn_frame = self.multiscale_tcn_frame(input, 0) # (N, 256, V, T)

        # 2.4 tcn output
        if self.add_idxfrm:
            #tcn_output = tcn_joint + tcn_frame + idx_frm # (N, 256, V, T)
            tcn_output = tcn_joint + idx_frm # (N, 256, V, T)
        else:
            tcn_output = tcn_joint #+ tcn_frame

        return tcn_output        

class unit_cascadeNet(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, args=None):
        super().__init__()
        num_classes = args.num_classes if args != None else 60

        self.block_idx_info = block_idx_info(args, out_jpt_channels=in_channels, out_frm_channels=out_channels)
        # gcn
        self.block_gcn = block_gcn(in_channels, out_channels, add_idxFrm=True)
        # tcn
        self.block_tcn = block_tcn(in_channels, out_channels, add_idxFrm=True)

        # classification
        self.unit_tcn = unit_tcn(out_channels * 2, out_channels * 2)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(out_channels * 2, num_classes)

    def forward(self, input):
        idx_jpt, idx_frm = self.block_idx_info()
        input = input  + idx_jpt

        gcn_output = self.block_gcn(input, idx_frm)
        tcn_output = self.block_tcn(input, idx_frm)
        tcn_gcn_input = torch.cat([tcn_output, gcn_output], dim=1) # (-1, 2 * out_channels, -1, -1)

        # classification
        output = self.unit_tcn(tcn_gcn_input) # (-1, 2 * out_channels, 1, -1)
        output = self.maxpool(output) # (-1, 2 * out_channels, 1, 1)
        output = torch.flatten(output, start_dim=1, end_dim=-1)
        output = self.fc(output)

        return output, tcn_gcn_input


class CascadeModel(nn.Module):
    def __init__(self, args):
        super(CascadeModel, self).__init__()
        V = 25

        ##### step 1: for input
        # idx info
        #self.block_idx_info = block_idx_info(args, out_jpt_channels=64, out_frm_channels=64*4)

        judge_dataType = args.judge_dataType if args != None else 0 # 0 for joint, 1 for bone
        mode_data = args.judge_dataMode if args != None else 2 # 0 for only joint|bone, 1 for t_joint|t_bone, 2 for joint + t_joint | bone + t_bone
        if judge_dataType == 0:
            self.block_input_info = block_joint_input(in_channels=3, out_channels=64, mode=mode_data)
        elif judge_dataType == 1:
            self.block_input_info = block_bone_input(in_channels=3, out_channels=64, mode=mode_data)

        # step 2: cascade net
        self.casNet_1 = unit_cascadeNet(64, 32, args)
        self.casNet_2 = unit_cascadeNet(64, 64, args)
        self.casNet_3 = unit_cascadeNet(128, 64, args)
        self.casNet_4 = unit_cascadeNet(128, 128, args)
        self.casNet_5 = unit_cascadeNet(256, 128, args)
        self.casNet_6 = unit_cascadeNet(256, 256, args)


    def forward(self, input):
        # step 1: for input
        # joint or bone
        input_info = self.block_input_info(input) # (N, 64, 25, T)

        # step 2: cascade net
        output_1, tcn_gcn_output_1 = self.casNet_1(input_info) # (N, 60) and (-1, 64, V, T)
        output_2, tcn_gcn_output_2 = self.casNet_2(tcn_gcn_output_1) # (N, 60) and (-1, 128, V, T)
        output_3, tcn_gcn_output_3 = self.casNet_3(tcn_gcn_output_2) # (N, 60) and (-1, 128, V, T)
        output_4, tcn_gcn_output_4 = self.casNet_4(tcn_gcn_output_3) # (N, 60) and (-1, 256, V, T)
        output_5, tcn_gcn_output_5= self.casNet_5(tcn_gcn_output_4) # (N, 60) and (-1, 256, V, T)
        output_6, tcn_gcn_output_6= self.casNet_6(tcn_gcn_output_5) # (N, 60) and (-1, 256, V, T)

        #return [output_1, output_2, output_3, output_4]
        return output_6

class CIGCN(nn.Module):
    def __init__(self, args):
        super(CIGCN, self).__init__()
        num_classes = args.num_classes if args != None else 60
        V = 17

        # 20211229
        self.with_attention = args.with_att
        self.with_block_tcn = args.with_block_tcn
        self.with_gcn_frm = args.with_gcn_frm
        self.with_gcn_tcn_para = args.with_gcn_tcn_para

        self.graph = Graph(labeling_mode='spatial')
        A = self.graph.A # (3, 25, 25)

        ##### step 1: for input
        # idx info
        self.block_idx_info = block_idx_info(args, out_jpt_channels=64, out_frm_channels=64*4)

        # joint + t_jnt; bone + t_bone
        # decide what data be used.
        judge_dataType = args.judge_dataType if args != None else 0 # 0 for joint, 1 for bone
        mode_data = args.judge_dataMode if args != None else 2 # 0 for only joint|bone, 1 for t_joint|t_bone, 2 for joint + t_joint | bone + t_bone
        if judge_dataType == 0:
            self.block_input_info = block_joint_input(in_channels=3, out_channels=64, mode=mode_data)
        elif judge_dataType == 1:
            self.block_input_info = block_bone_input(in_channels=3, out_channels=64, mode=mode_data)
              
        ##### 1 for GCN module
        #self.block_gcn = block_gcn(add_idxFrm=True)
        self.block_gcn = block_gcn_debug(add_idxFrm=True, attention=self.with_attention, gcn_frm=self.with_gcn_frm) # 20211229, adaptive by attention and gcn_frame for debyg
        #self.block_gcn = block_gcn(add_idxFrm=True, A=A)

        ##### 2 for TCN module
        if self.with_block_tcn:
            if self.with_gcn_tcn_para:
                self.block_tcn = block_tcn(add_idxFrm=True)
                in_c = 512
            else:
                self.block_tcn = block_tcn(add_idxFrm=True, in_channels=256)
                in_c = 256

        else:
            in_c = 256

        ##### 3 for classification
        self.module_tcn = unit_tcn(in_c, 512)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

        # initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                '''
                n = m.out_channels
                k1 = m.kernel_size[0]
                k2 = m.kernel_size[1]
                n = n * k1 * k2
                m.weight.data.normal_(0, math.sqrt(2. / n))
                '''

                ''' from sgn
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                '''

                conv_init(m)
            if isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, input):

        # step 1: process input info
        idx_jpt, idx_frm = self.block_idx_info() # (N, 64, V, T) and (N, 64*4, V, T)

        # joint or bone
        input_info = self.block_input_info(input) # (N, 64, 25, T)
        
        input = torch.cat([input_info, idx_jpt], 1)  # (N, 64*2, V, T)

        ##### 1 for GCN module
        gcn_output = self.block_gcn(input, idx_frm)# (N, 256, V, T)
        #gcn_output_g1 = self.block_gcn_g1(input_g1, idx_frm) # (N,256, 11, T)

        if self.with_block_tcn:
            if self.with_gcn_tcn_para:
                ##### 2 for TCN module
                tcn_output = self.block_tcn(input, idx_frm) # (N, 256, V, T)
                tcn_gcn_input = torch.cat([tcn_output, gcn_output], dim=1) # (N, 512, V, T)
            else:
                tcn_output = self.block_tcn(gcn_output, idx_frm)
                tcn_gcn_input = tcn_output # (N, 256, V, T)

        else:
            tcn_gcn_input = gcn_output # (N, 256, V, T)

        ##### 3 classification
        output = self.module_tcn(tcn_gcn_input) # (N, 512, 1, 20)
        #output_g1 = self.module_tcn_g1(gcn_output_g1) # (N, 512, 1, 20)
        output = self.maxpool(output) # + output_g1) # (N, 512, 1, 1)

        ## flatten
        output = torch.flatten(output, start_dim=1, end_dim=-1) # (N, 512) 
        output = self.fc1(output)

        return output


class CIGCN_before20220712(nn.Module):
    def __init__(self, args):
        super(CIGCN_before20220712, self).__init__()
        num_classes = args.num_classes if args != None else 60
        V = 25

        # 20211229
        self.with_attention = args.with_att
        self.with_block_tcn = args.with_block_tcn
        self.with_gcn_frm = args.with_gcn_frm
        self.with_gcn_tcn_para = args.with_gcn_tcn_para

        self.graph = Graph(labeling_mode='spatial')
        A = self.graph.A # (3, 25, 25)

        ##### step 1: for input
        # idx info
        self.block_idx_info = block_idx_info(args, out_jpt_channels=64, out_frm_channels=64*4)

        # joint + t_jnt; bone + t_bone
        # decide what data be used.
        judge_dataType = args.judge_dataType if args != None else 0 # 0 for joint, 1 for bone
        mode_data = args.judge_dataMode if args != None else 2 # 0 for only joint|bone, 1 for t_joint|t_bone, 2 for joint + t_joint | bone + t_bone
        if judge_dataType == 0:
            self.block_input_info = block_joint_input(in_channels=3, out_channels=64, mode=mode_data)
        elif judge_dataType == 1:
            self.block_input_info = block_bone_input(in_channels=3, out_channels=64, mode=mode_data)
              
        ##### 1 for GCN module
        #self.block_gcn = block_gcn(add_idxFrm=True)
        self.block_gcn = block_gcn_debug(add_idxFrm=True, attention=self.with_attention, gcn_frm=self.with_gcn_frm) # 20211229, adaptive by attention and gcn_frame for debyg
        #self.block_gcn = block_gcn(add_idxFrm=True, A=A)

        ##### 2 for TCN module
        if self.with_block_tcn:
            if self.with_gcn_tcn_para:
                self.block_tcn = block_tcn(add_idxFrm=True)
                in_c = 512
            else:
                self.block_tcn = block_tcn(add_idxFrm=True, in_channels=256)
                in_c = 256

        else:
            in_c = 256

        ##### 3 for classification
        self.module_tcn = unit_tcn(in_c, 512)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

        # initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                '''
                n = m.out_channels
                k1 = m.kernel_size[0]
                k2 = m.kernel_size[1]
                n = n * k1 * k2
                m.weight.data.normal_(0, math.sqrt(2. / n))
                '''

                ''' from sgn
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                '''

                conv_init(m)
            if isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, input):

        # step 1: process input info
        idx_jpt, idx_frm = self.block_idx_info() # (N, 64, V, T) and (N, 64*4, V, T)

        # joint or bone
        input_info = self.block_input_info(input) # (N, 64, 25, T)
        
        input = torch.cat([input_info, idx_jpt], 1)  # (N, 64*2, V, T)

        ##### 1 for GCN module
        gcn_output = self.block_gcn(input, idx_frm)# (N, 256, V, T)
        #gcn_output_g1 = self.block_gcn_g1(input_g1, idx_frm) # (N,256, 11, T)

        if self.with_block_tcn:
            if self.with_gcn_tcn_para:
                ##### 2 for TCN module
                tcn_output = self.block_tcn(input, idx_frm) # (N, 256, V, T)
                tcn_gcn_input = torch.cat([tcn_output, gcn_output], dim=1) # (N, 512, V, T)
            else:
                tcn_output = self.block_tcn(gcn_output, idx_frm)
                tcn_gcn_input = tcn_output # (N, 256, V, T)

        else:
            tcn_gcn_input = gcn_output # (N, 256, V, T)

        ##### 3 classification
        output = self.module_tcn(tcn_gcn_input) # (N, 512, 1, 20)
        #output_g1 = self.module_tcn_g1(gcn_output_g1) # (N, 512, 1, 20)
        output = self.maxpool(output) # + output_g1) # (N, 512, 1, 1)

        ## flatten
        output = torch.flatten(output, start_dim=1, end_dim=-1) # (N, 512) 
        output = self.fc1(output)

        return output


class CIGCN_before_1228(nn.Module):
    def __init__(self, args):
        super(CIGCN, self).__init__()
        num_classes = args.num_classes if args != None else 60
        V = 25

        self.graph = Graph(labeling_mode='spatial')
        A = self.graph.A # (3, 25, 25)

        ##### step 1: for input
        # idx info
        self.block_idx_info = block_idx_info(args, out_jpt_channels=64, out_frm_channels=64*4)

        # joint + t_jnt; bone + t_bone
        # decide what data be used.
        judge_dataType = args.judge_dataType if args != None else 0 # 0 for joint, 1 for bone
        mode_data = args.judge_dataMode if args != None else 2 # 0 for only joint|bone, 1 for t_joint|t_bone, 2 for joint + t_joint | bone + t_bone
        if judge_dataType == 0:
            self.block_input_info = block_joint_input(in_channels=3, out_channels=64, mode=mode_data)
        elif judge_dataType == 1:
            self.block_input_info = block_bone_input(in_channels=3, out_channels=64, mode=mode_data)
              
        ##### 1 for GCN module
        self.block_gcn = block_gcn(add_idxFrm=True)
        #self.block_gcn = block_gcn(add_idxFrm=True, A=A)

        ##### 2 for TCN module
        self.block_tcn = block_tcn(add_idxFrm=True)

        ##### 3 for classification
        self.module_tcn = unit_tcn(512, 512)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

        # initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                '''
                n = m.out_channels
                k1 = m.kernel_size[0]
                k2 = m.kernel_size[1]
                n = n * k1 * k2
                m.weight.data.normal_(0, math.sqrt(2. / n))
                '''

                ''' from sgn
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                '''

                conv_init(m)
            if isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, input):

        # step 1: process input info
        idx_jpt, idx_frm = self.block_idx_info() # (N, 64, V, T) and (N, 64*4, V, T)

        # joint or bone
        input_info = self.block_input_info(input) # (N, 64, 25, T)
        
        input = torch.cat([input_info, idx_jpt], 1)  # (N, 64*2, V, T)

        ##### 1 for GCN module
        gcn_output = self.block_gcn(input, idx_frm)# (N, 256, V, T)
        #gcn_output_g1 = self.block_gcn_g1(input_g1, idx_frm) # (N,256, 11, T)

        ##### 2 for TCN module
        tcn_output = self.block_tcn(input, idx_frm) # (N, 256, V, T)
        #tcn_output = self.block_tcn(input, idx_frm, adaA_joint, adaA_frame) # (N, 256, V, T)

        # flip
        #tcn_output = torch.flip(tcn_output, dims=[-1])

        ##### 3 classification
        tcn_gcn_input = torch.cat([tcn_output, gcn_output], dim=1) # (N, 512, V, T)
        #tcn_gcn_input = gcn_output + tcn_output # (N, 256, V, T)
        #tcn_gcn_input = torch.cat([gcn_output, idx_frm], dim=1) # (N, 512, V, T)

        output = self.module_tcn(tcn_gcn_input) # (N, 512, 1, 20)
        #output_g1 = self.module_tcn_g1(gcn_output_g1) # (N, 512, 1, 20)
        output = self.maxpool(output) # + output_g1) # (N, 512, 1, 1)

        ## flatten
        output = torch.flatten(output, start_dim=1, end_dim=-1) # (N, 512) 
        output = self.fc1(output)

        return output

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
   

if __name__ == "__main__":
    print ('s')

    model = CIGCN(60, True, 20, True)
    total = get_n_params(model)
    #print(model)
    print('The number of parameters: ', total)  # by gzb: 3379348 parameters
    print('The modes is:', model)  # by gzb: cigcn

