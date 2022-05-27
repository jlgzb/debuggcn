from collections import OrderedDict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

#refererence 'https://www.freesion.com/article/8735861895/'
# 1
class SELayer(nn.Module):
    '''
    func: achieve channels Attention.  "Squeeze-and-Excitation Networks"
    parameters:
        input.size: (batch, channel, w, h)
        reduction: default 4. performe in FC: channel --> channel//reduction --> channel
    '''
    def __init__(self, channels, inter_channels):
        super(SELayer, self).__init__()

        #self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.AdaptiveMaxPool2d(1)

        self.fc_in = nn.Linear(channels, inter_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(inter_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        
    def forward(self, input):
        '''
        input.size == output.size 
        '''
        N, C, _, _ = input.size()

        out = self.pool(input).view(N, C).contiguous() # (N, C)

        # compute attention
        out = self.fc_in(out)
        out = self.relu(out)
        out = self.fc_out(out)
        out = self.sigmoid(out)

        # combine with input
        w = out.view(N, C, 1, 1).contiguous()
        out = input * w.expand_as(input) # (N, C, V, T)

        return out # (N, C, V, T)

# 2
class ChannelAttention(nn.Module):
    r"""
    samiliar as 'SMlayer' model
    input.size: N, C, V, T
    """
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttention, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size = 1, bias = False),
            )

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def forward(self, input):
        r"""
        input: (N, C, V, T)
        output: (N, C, V, T)
        """

        out_avg = self.avgpool(input) # shape (N, C, 1, 1)
        out_max = self.maxpool(input)

        # compute attention
        out_avg = self.sharedMLP(out_avg)
        out_max = self.sharedMLP(out_max)
        out = out_avg + out_max  # (N, C, 1, 1)

        # combine with input
        w = self.sigmoid(out) # weight # (N, C, 1, 1)
        out = input * w.expand_as(input) # (N, C, V, T)

        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class SpatialAttention(nn.Module):
    r"""
    func: achieve spatial Attention. 
    parameters:
        kernel_size: can be 3,5,7
        input_size: (batch_size, channels, w, h)
    """
    def __init__(self, kernel_size = 3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,5,7), "kernel size must be 3 or 7"

        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def forward(self, input):
        out_avg = torch.mean(input, dim=1, keepdim=True) # (batch, 1, w, h)
        out_max, _ = torch.max(input, dim=1, keepdim=True) # (batch, 1, w, h)

        out = torch.cat([out_avg, out_max], dim=1) # (batch, 2, w, h)
        out = self.conv(out) # (batch, 1, w, h)

        w = self.sigmoid(out)
        out = input * w

        return out
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class CBAtt_Res(nn.Module):
    r"""
    # convolutional block attention module (CBAM)

    func: channel_attention + spatial_attention + resnet
    parameters:
         input.size = (batch, in_channels, w, h);
        out_channels: 
        kernel_size: default 3, can be select from [3,5,7]
        stride: default 2; which perform: out.size --> (batch, out_channels, w/stride, h/stride).
                generally: out_channels = in_channels * stride
        reduction: default 4. for channel_atten of FC: in_channels --> in_channels//reduction --> in_channels
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, reduction=4):
        
        super(CBAtt_Res,self).__init__()
        self.reduction = reduction
        self.padding = kernel_size // 2

        #h/2, w/2
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=kernel_size,
                               stride=stride, 
                               padding = self.padding,
                               bias=True)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

        self.att_channel = ChannelAttention(out_channels, reduction=self.reduction)
        self.att_spatial = SpatialAttention(kernel_size=kernel_size)

        #h/2, w/2
        self.downsample = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=self.padding), # (batch, in_channels, w/stride, h/stride)
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias = True)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        r"""
        input: (N, C, V, T)
        if stride=1:
            output: (N, C, V, T)
        elif stride=2:
            output: (N, -1, V//2, T//2)
        """
        residual = input

        out = self.conv1(input) # (batch, out_channels, w/stride, h/stride)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.att_channel(out)
        out = self.att_spatial(out) # (batch, out_channels, w/stride, h/stride)

        # res
        residual = self.downsample(residual) # (batch, out_channels, w/stride, h/stride)
        out += residual

        out = self.relu(out) # (batch, out_channels, w/stride, h/stride)

        return out

# 3
class SKEConv(nn.Module):
    r"""
    # Selective Kernel Networks(SKE) Attention
    N, C, V, T
    func: comsist of Spit + Fuse + Select modules
    parameters:
        in_channels: input.size(0)
        num_kernels: Split阶段. 使用不同大小的卷积核(M个)对input进行卷积，得到M个分支，默认2;
        G: 在卷积过程中使用分组卷积，分组个数为G, default 2.可以减小参数量;
        stride: 1 (default). split卷积过程中的stride,也可以选2，降低输入输出的w,h;
        L: 32 (default); 
        reduction: 默认2，压缩因子; 在线性部分压缩部分，输出特征d = max(L, in_channels / reduction);
    """
    def __init__(self,in_channels, num_kernels=2, G=2, stride=1, L=32, reduction=2):
        super(SKEConv,self).__init__()
        self.M = 2
        self.in_channels = in_channels

        self.convs = nn.ModuleList([])

        for i in range(num_kernels):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size = 3 + i*2,
                    stride=stride, padding=1+i, groups=G),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)                
                )
            )

        self.d = max(int(in_channels/reduction), L)
        self.fc = nn.Linear(in_channels, self.d)

        self.fcs = nn.ModuleList([])

        for i in range(num_kernels):
            self.fcs.append(nn.Linear(self.d, in_channels))

        self.softmax = nn.Softmax(dim = 1)     

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        r"""
        input: (N, C, W, H)
        output: (N, C, W, H)
        """

        # module: U
        for i, conv in enumerate(self.convs):
            out = conv(input).unsqueeze_(dim=1)  # (batch, 1, in_channels, w, h)
            if i == 0:
                outs = out
            else:
                outs = torch.cat([outs, out], dim=1) # (batch, num_kernels, in_channels, w, h)

        fea_u = torch.sum(outs, dim = 1) # (batch, in_channels, w, h)
        fea_s = fea_u.mean(-1).mean(-1) # (batch, in_channels)
        fea_z = self.fc(fea_s)  # size = (batch, d)

        # module: soft attention
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1) # (batch, 1, in_channels)

            # record vector
            if i == 0:
                attention_vectors = vector
            else:
                #return shape: (batch, num_kernels, in_channels)
                attention_vectors = torch.cat([attention_vectors,  vector], dim=1)

        attention_vectors = self.softmax(attention_vectors) # (batch, num_kernels, in_channels)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1) # (batch, num_kernels, in_channels, 1, 1)
        output = (outs * attention_vectors).sum(dim=1) # (batch, in_channels, w, h)

        return output

# 4
class SelfAtt_Spatial(nn.Module):
    r""" 
    N, C, V, T
    func: Self attention Spatial Layer 自注意力机制.通过类似Transformer中的Q K V来实现
    inputs:
        in_channels: 输入的通道数
        out_channels: 在进行self attention时生成Q,K矩阵的列数, 一般默认为in_dim//8
    """
    def __init__(self, in_channels, out_channels):
        super(SelfAtt_Spatial, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def forward(self, input):
        r"""
            inputs :
                input : input feature maps(N, C, H, W)
            returns :
                out : self attention value + input feature; shape (N, C, H, W)
                attention: (N, H*W, H*W)
        """
        N, C, H, W = input.size()

        # proj_query中的第i行表示第i个像素位置上所有通道的值
        proj_query = self.conv_query(input) # (N, out_channels, H, W)
        proj_query = proj_query.view(N, -1, H * W).permute(0, 2, 1).contiguous() # (N, H*W, -1)

        # proj_key中的第i行表示第i个通道上所有像素的值
        proj_key = self.conv_key(input)
        proj_key = proj_key.view(N, -1, H * W).contiguous() # (N, -1, H*W)

        proj_value = self.conv_value(input) # (N, C, H, W)
        proj_value = proj_value.view(N, -1, H*W).contiguous() # (N, -1, H*W)   

        # energy中的第(i,j)是将proj_query中的第i行与proj_key中的第j列点乘得到
        # energy中第(i,j)位置的元素是指输入特征图(input)第j个元素对第i个元素的影响 by gzb: ???
        # 从而实现全局上下文任意两个元素的依赖关系
        # by gzb: the relationship between different pixel though all channels
        energy =  torch.bmm(proj_query, proj_key) # transpose check # (N, H*W, H*W)

        # 对行的归一化, 即每行的所有列加起来为1
        # 对于(i,j)位置, 可理解为第j位置对i位置的权重, 所有的j对i位置的权重之和为1
        attention = self.softmax(energy) # (N, H*W, H*W)
        temp_attention = attention.permute(0, 2, 1).contiguous()    

        att = torch.bmm(proj_value, temp_attention) # (N, -1, H*W)
        att = att.view(N, -1, H, W).contiguous() # (N, out_channels, H, W)
        att = self.conv_out(att) # (N, in_channels, H, W)
        
        # skip connection; Gamma is the param that need to be learned
        out = self.gamma * att + input

        return out, attention

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class SelfAtt_channel(nn.Module):
    r""" 
    func: Self attention Channel Layer 自注意力机制.通过类似Transformer中的Q K V来实现
    inputs:
        in_channels: 输入的通道数
        out_channels: 在进行self attention时生成Q,K矩阵的列数, 默认可选取为：in_dim
        
    """
    def __init__(self, in_channels, out_channels):
        super(SelfAtt_channel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # same as SelfAtt_Spatial
        self.conv_query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        # new added for channels attention
        self.conv_input = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)

        self.init_weights()

    def forward(self, input):
        r"""
            inputs :
                input : input feature maps(N, C, H, W)
            returns :
                out : self attention value + input feature; shape (N, C, H, W)
                attention: (N, H*W, H*W)
        """
        N, C, H, W = input.size()
        
        #  proj_query中的第i行表示第i个像素位置上所有通道的值
        proj_query = self.conv_query(input) # (N, out_channels, H, W)
        proj_query = proj_query.view(N, -1, H * W).contiguous() # (N, -1, H*W)

        # proj_key中的第i行表示第i个通道上所有像素的值
        proj_key = self.conv_key(input)
        proj_key = proj_key.view(N, -1, H * W).permute(0,2,1).contiguous() # (N, H*W, -1)

        proj_value = self.conv_value(input) # (N, -1, H, W)
        proj_value = proj_value.view(N, -1, H*W).contiguous() # (N, -1, H*W)

        # energy中的第(i,j)是将proj_query中的第i行与proj_key中的第j列点乘得到
        # energy中第(i,j)位置的元素是指输入特征图(input)第j个元素对第i个元素的影响 by gzb: ???
        # 从而实现全局上下文任意两个元素的依赖关系
        # by gzb: the relationship between different pixel though all channels
        energy =  torch.bmm(proj_query, proj_key) # transpose check # (N, out_channels, out_channels)

        # 对行的归一化, 即每行的所有列加起来为1
        # 对于(i,j)位置, 可理解为第j位置对i位置的权重, 所有的j对i位置的权重之和为1
        attention = self.softmax(energy) # (N, out_channels, out_channels)

        out = torch.bmm(attention, proj_value) # (N, out_channels, H*W)
        out = out.view(N, self.out_channels, H, W).contiguous()

        # skip connect, learng Gamma
        out = self.gamma * out + self.conv_input(input)

        return out, attention

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 5
class NonLocalBlockND(nn.Module):
    """
    func: 非局部信息统计的注意力机制
    inputs: 
        in_channels: 输入的通道数
        inter_channels: 
            生成attention时Conv的输出通道数， 一般为in_channels//2.
            如果为None, 则自动为in_channels//2
        dimension: 默认2.可选为[1,2,3]，
            1: 输入为size = [batch, in_channels, width] or [batch,time_steps,seq_length]，可表示时序数据
            2: 输入size = [batch, in_channels, width, height], 即图片数据
            3: 输入size = [batch, time_steps, in_channels, width, height]，即视频数据
                    
        down_size: 默认True, 是否在Attention过程中对input进行size降低，即 w,h = w//2, h//2               
        bn_layer: 默认True
    """
    def __init__(self, in_channels, inter_channels=None, dimension=2, down_size=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.dimension = dimension
        self.down_size = down_size
        self.bn_layer = bn_layer

        self.get_layer()

        self.conv1 = self.conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                kernel_size=1, stride=1, padding=0)

        # set param
        if self.bn_layer:
            self.W = nn.Sequential(
                self.conv_nd(
                        in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                self.bn(self.in_channels)
            )
        else:
            self.W = self.conv_nd(
                    in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0)

        self.theta = self.conv_nd(
                    in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0)
                             
        self.phi = self.conv_nd(
                    in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0)

        # down size of feature map
        if down_size:
            self.conv1 = nn.Sequential(self.conv1, self.max_pool_layer)
            self.phi = nn.Sequential(self.phi, self.max_pool_layer)

        self.init_weights()

    def forward(self, input):
        r"""
            inputs :
                size : (N, C, W, H)
            outputs :
                size : (N, C, W, H)

        """
        # if dimension == 3 , F = w*h*t ; if sub_sample: F1 = (w//2) * (h//2) * t , else: F1 = F
        # if dimension == 2 , F = w*h
        # if dimension == 1 , F = w 
        # C0 = in_channels; C1 = inter_channels
        

        batch_size = input.size(0) # N = batch_size 

        conv_x = self.conv1(input).view(batch_size, self.inter_channels, -1) # (N, C1, F)
        conv_x = conv_x.permute(0, 2, 1).contiguous() # (N, F, C1)

        theta_x = self.theta(input).view(batch_size, self.inter_channels, -1) # (N, C1, F)
        theta_x = theta_x.permute(0, 2, 1) # (N, F, C1)

        phi_x = self.phi(input).view(batch_size, self.inter_channels, -1) # (N, C1, F)

        out = torch.matmul(theta_x, phi_x) # (N, F, F)
        out = F.softmax(out, dim=-1) # (N, F, F)

        out = torch.matmul(out, conv_x) # (N, F, C1)
        out = out.permute(0, 2, 1).contiguous() # (N, C1, F)

        size = [batch_size, self.inter_channels] + list(input.size()[2:]) # [N, C1] + [w, h, t]
        out = out.view(size).contiguous() # (N, C1, W, H)

        out = self.W(out) + input

        return out

    def get_layer(self):
        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.in_channels == 0:
                self.in_channels = 1

        if self.dimension == 3:
            self.conv_nd = nn.Conv3d
            self.max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.bn = nn.BatchNorm3d
        elif self.dimension == 2:
            self.conv_nd = nn.Conv2d
            self.max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.bn = nn.BatchNorm2d
        else:
            self.conv_nd = nn.Conv1d
            self.max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            self.bn = nn.BatchNorm1d

    def init_weights(self):
        if self.bn_layer:
            # init with constant
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

# reference:
# https://github.com/xmu-xiaoma666/External-Attention-pytorch
# 1
class ExternalAtt(nn.Module):
    r"""
    func: ExternalAttention
    shape of input: 
        (N, seq_length, input_size)
        C0: in_channels (input_size); 
        C1: inter_channels
    """

    def __init__(self, in_channles, inter_channels=64):
        super(ExternalAtt, self).__init__()

        self.fc1 = nn.Linear(in_channles, inter_channels, bias=False)
        self.fc2 = nn.Linear(inter_channels, in_channles, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        out = self.fc1(input) # (N, seq_length, C1)
        out = self.softmax(out) # (N, seq_length, C1)
        out_sum = torch.sum(out, dim=2, keepdim=True) # (N, seq_length, 1)
        out = out / out_sum # (N, seq_length, C1)
        out = self.fc2(out) # (N, seq_length, C0)
        return out

# 2
class SDPAtt(nn.Module):
    '''
    ori name: ScaledDotProductAttention
    Scaled dot-product attention
    self attention
    '''
    def __init__(self, channels, dim_key, dim_value, num_heads, dropout=.1):
        '''
        shape:
            input: (batch_size, nq, channels)
            output: (batch_size, nq, channels)
        param: 
            channels: Output dimensionality of the model
            dim_key: Dimensionality of queries and keys
            dim_value: Dimensionality of values
            num_head: Number of heads
        '''
        super(SDPAtt, self).__init__()

        self.channels = channels
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.num_heads = num_heads

        self.fc_query = nn.Linear(channels, num_heads * dim_key)
        self.fc_key = nn.Linear(channels, num_heads * dim_key)
        self.fc_value = nn.Linear(channels, num_heads * dim_value)
        self.fc_out = nn.Linear(num_heads * dim_value, channels)

        self.dropout=nn.Dropout(dropout)

        self.init_weights()

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        param: 
            queries: Queries (batch_size, nq, channels)
            keys: Keys (batch_size, nk, channels)
            values: Values (b_s, nk, channels)
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        return;
        '''
        bs, nq = queries.shape[:2]
        nk = keys.shape[1]

        query = self.fc_query(queries) # (bs, nq, nh * dk)
        query = query.view(bs, nq, self.num_heads, self.dim_key).permute(0, 2, 1, 3) # (bs, nh, nq, dk)

        key = self.fc_key(keys) # (bs, nk, nh * dk)
        key = key.view(bs, nk, self.num_heads, self.dim_key).permute(0, 2, 3, 1) # (bs, nh, dk, nk)

        value = self.fc_value(values) # (bs, nk, nh * dk)
        value = value.view(bs, nk, self.num_heads, self.dim_value).permute(0, 2, 1, 3) # (bs, nh, nk, dv)

        att = torch.matmul(query, key) # (bs, nh, nq, nk)
        att = att / np.sqrt(self.dim_key) # (bs, nh, nq, nk)

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, value) # (bs, nh, nq, nk) * (bs, nh, nk, dv) == (bs, nh, nq, dv)
        out = out.permute(0, 2, 1, 3).contiguous() # (bs, nq, nh, dv)
        out = out.view(bs, nq, self.num_heads * self.dim_value)
        out = self.fc_out(out)

        return out    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 3
class SimSDPAtt(nn.Module):
    '''
    Scaled dot-product attention: 
    ori name: SimplifiedScaledDotProductAttention
    different between SDPAtt and SimSDPAtt are that:
        do not perform self.fc for queries, keys, and values, respectively.
    '''
    def __init__(self, channels, num_heads, dropout=.1):
        '''
        shape:
            input: (bs, nq, channels)
            output: (bs, nq, channels)
        param:
            channels: Output dimensionality of the model
            dim_key: Dimensionality of queries and keys
            dim_value: Dimensionality of values
            dim_heads: Number of heads
        '''
        super(SimSDPAtt, self).__init__()
        self.channels = channels
        self.dim_key = channels // num_heads
        self.dim_value = channels // num_heads
        self.num_heads = num_heads

        self.fc_out = nn.Linear(num_heads * self.dim_value, channels)
        self.dropout=nn.Dropout(dropout)

        self.init_weights()

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        param: 
            queries: Queries (batch_size, nq, channels)
            keys: Keys (batch_size, nk, channels)
            values: Values (b_s, nk, channels)
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        return;
        '''
        bs, nq = queries.shape[:2]
        nk = keys.shape[1]

        query = queries.view(bs, nq, self.num_heads, self.dim_key).permute(0, 2, 1, 3) # (bs, nh, nq, dk)
        key = keys.view(bs, nk, self.num_heads, self.dim_key).permute(0, 2, 3, 1) # (bs, nh, dk, nk)
        value = values.view(bs, nk, self.num_heads, self.dim_value).permute(0, 2, 1, 3) # (bs, nh, nk, dv)

        att = torch.matmul(query, key) # (bs, nh, nq, nk)
        att = att / np.sqrt(self.dim_key) # (bs, nh, nq, nk)

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, value) # (bs, nh, nq, nk) * (bs, nh, nk, dv) == (bs, nh, nq, dv)
        out = out.permute(0, 2, 1, 3).contiguous() # (bs, nq, nh, dv)
        out = out.view(bs, nq, self.num_heads * self.dim_value)
        out = self.fc_out(out)

        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


# 4 equal to SELayer
# class SEAttention(nn.Module)...

# 5 equal to SKEConv
class SKAtt(nn.Module):
    r"""
    "Selective Kernel Networks"
    """
    def __init__(self, channels=512, kernels=[1,3,5,7], reduction=16, group=1, L=32):
        super(SKAtt, self).__init__()

        self.channels = channels

        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    OrderedDict([
                        ('conv', nn.Conv2d(self.channels, self.channels, kernel_size=k, padding=k //2, groups=group)),
                        ('bn', nn.BatchNorm2d(self.channels)),
                        ('relu', nn.ReLU())
                    ])
                )
            )

        self.d = max(L, self.channels//reduction)
        self.fc = nn.Linear(self.channels, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, self.channels))

        self.softmax = nn.Softmax(dim=0)

        self.init_weights()

    def forward(self, input):
        r"""
        input: (N, C, W, H)
        output: (N, C, W, H)
        """
        list_outs = []
        for convNet in self.convs:
            list_outs.append(convNet(input))
        outs = torch.stack(list_outs, dim=0) # (num_kernel, N, self.channels, W, H)

        # module: U
        fea_u = torch.sum(outs, dim=0)  # (N, self.channels, W, H)

        # reduction channel
        fea_s = fea_u.mean(-1).mean(-1) # (N, self.channels)
        fea_z = self.fc(fea_s) # (N, self.d)

        weights = []
        for fc in self.fcs:
            weight = fc(fea_z) # (N, self.channels)
            weight = weight.view(input.size(0), self.channels, 1, 1).contiguous() # (N, self.channels, 1, 1)
            weights.append(weight)  # (num_kernel, N, self.channels, 1, 1)
        
        atention_weights = torch.stack(weights, dim=0) # (num_kernel, N, self.channels, 1, 1)
        atention_weights = self.softmax(atention_weights) # (num_kernel, N, self.channels, 1, 1)

        # fuse
        output = (atention_weights * outs).sum(dim=0) # (N, self.channels, 1, 1)

        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 6 equal to CBAtt_Res

# 7 BAMBloack: "BAM: Bottleneck Attention Module"
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)

class ChannelAtt(nn.Module):
    '''
    shape:
        input: (N, C, W, H)
        output: (N, C, W, H)
    '''
    def __init__(self, channels, reduction=16, num_layers=3):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        gate_channels = [channels]
        gate_channels += [channels // reduction] * num_layers
        gate_channels += [channels] # [channels, s, s, s, channels], s = channels // reduction

        self.conv = nn.Sequential()
        self.conv.add_module('flatten', Flatten())
        for i in range(len(gate_channels) - 2): # 0-based
            self.conv.add_module('fc%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]))
            self.conv.add_module('bn%d'%i, nn.BatchNorm1d(gate_channels[i+1]))
            self.conv.add_module('relu%d'%i, nn.ReLU())
        
        # the last conv2d of self.conv not need bn and relu
        self.conv.add_module('fc_last', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, input):
        out = self.avg_pool(input) # (N, C, 1, 1)
        out = self.conv(out) # (N, -1)
        out = out.unsqueeze(-1).unsqueeze(-1) # (N, -1, 1, 1)
        out = out.expand_as(input)
        return out

class SpatialAtt(nn.Module):
    '''
    shape:
        input: (N, C, W, H)
        output: (N, 1, W, H)
    '''
    def __init__(self, channels, reduction=16, num_layers=3, dia_val=2):
        super(SpatialAtt, self).__init__()
        self.inter_channels = channels // reduction
        self.conv = nn.Sequential()
        self.conv.add_module('conv_reduce1', nn.Conv2d(channels, self.inter_channels, kernel_size=1))
        self.conv.add_module('bn_reduce1', nn.BatchNorm2d(self.inter_channels))
        self.conv.add_module('relu_reduce1', nn.ReLU())
        for i in range(num_layers):
            self.conv.add_module('conv%d'%i, nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=3, padding=1, dilation=dia_val))
            self.conv.add_module('bn%d'%i, nn.BatchNorm2d(self.inter_channels))
            self.conv.add_module('relu%d'%i, nn.ReLU())

        self.conv.add_module('conv_last', nn.Conv2d(self.inter_channels, 1, kernel_size=1))

    def forward(self, input):
        out = self.conv(input) # (N, 1, W, H)
        out = out.expand_as(input)
        return out
        
class BAMBlock(nn.Module):
    '''
    shape:
        input: (N, C, W, H)
        output: (N, C, W, H)
    '''
    def __init__(self, channels=512,reduction=16, dia_val=2):
        super(BAMBlock, self).__init__()
        self.att_channels = ChannelAtt(channels=channels, reduction=reduction)
        self.att_spatial = SpatialAtt(channels=channels, reduction=reduction, dia_val=dia_val)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out_c = self.att_channels(input) # (N, C, W, H)
        out_s = self.att_spatial(input) # (N, 1, W, H)
        att = self.sigmoid(out_s + out_c)

        output = input + att*input

        return output

# 8
class ECAtt(nn.Module):
    '''
    "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    '''
    def __init__(self, kernel_size=3):
        super(ECAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def forward(self, input):
        out = self.avg_pool(input) # (N, C, 1, 1)
        out = out.squeeze(-1).permute(0, 2, 1).contiguous() # (N, 1, C)
        out = self.conv(out)
        out = self.sigmoid(out) # (N, 1, C)
        out = out.permute(0, 2, 1).unsqueeze(-1).contiguous() # (N, C, 1, 1)
        out = input * out.expand_as(input) # (N, C, W, H)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 9
#utilize SDPAtt and SimSDPAtt
class PositionAtt(nn.Module):
    def __init__(self,channels=512, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.att_sdp = SDPAtt(channels, dim_key=channels, dim_value=channels, num_heads=1)

    def forward(self, input):
        '''
        shape:
            input: (N, C, H, W)
            output: (N, H*W, C)
        '''
        N, C, W, H = input.shape
        out = self.conv(input) # (N, C, H, W)
        out = out.view(N, C, -1).permute(0, 2, 1).contiguous() # (N, -1, C)
        out = self.att_sdp(out, out, out) # conv for C
        return out # (N, H*W, C)
    
class ChannelAtt_DualAtt(nn.Module):
    def __init__(self, channels=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.att_simSDP = SimSDPAtt(channels=W*H, num_heads=1)

    def forward(self, input):
        '''
        shape:
            input: (N, C, H, W)
            output: (N, C, H*W)
        '''
        N, C, W, H = input.shape
        out = self.conv(input) # (N, C, H, W)
        out = out.view(N, C, -1).contiguous()
        out = self.att_simSDP(input, input, input) # conv for H*W
        return out

class DualAtt(nn.Module):
    '''
    "Dual Attention Network for Scene Segmentation"
    func: 
        1: perform conv for C by PositionAtt (SDPAtt)
        2: perform conv for H*W by ChannelAtt_DualAtt (ChannelAtt_DualAtt)
    '''
    def __init__(self, channels=512, kernel_size=3, H=25, W=20):
        super(DualAtt, self).__init__()
        self.att_pos = PositionAtt(channels, kernel_size=kernel_size)
        self.att_cha = ChannelAtt_DualAtt(channels, kernel_size=kernel_size, H=H, W=W)

    def forward(self, input):
        N, C, H, W = input.shape
        out_pos = self.att_pos(input) # (N, H*W, C)
        out_pos = out_pos.permute(0, 2, 1).view(N, C, H, W)

        out_cha = self.att_cha(input) # (N, C, H*W)
        out_cha = out_cha.view(N, C, H, W)

        out = out_pos + out_cha
        return out 

# 10
class PyramidSplitAtt(nn.Module):
    '''
    "EPSANet: An Efficient Pyramid Split Attention Block on Convolutional Neural Network"
    '''
    def __init__(self, channels=512, reduction=4, split=4):
        super(PyramidSplitAtt, self).__init__()
        self.split = split

        #'''
        #self.convs = []
        self.convs = nn.ModuleList([])
        for i in range(split):
            self.convs.append(nn.Conv2d(channels//split, channels//split, kernel_size=1+2*(i+1), padding=i+1))
        #'''
        
        #self.conv_blocks = []
        self.conv_blocks = nn.ModuleList([])
        for i in range(split):
            self.conv_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels//split, channels//(split*reduction), kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channels//(split*reduction), channels//split, kernel_size=1, bias=False),
                nn.Sigmoid()
            ))

        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def forward(self, input):
        '''
        shape:
            input: (N, C, H, W)
            output: (N, C, H, W)
        '''
        N, C, H, W = input.shape

        # step 1: split conv for C
        out_1 = input.view(N, self.split, C//self.split, H, W)
        for idx, conv in enumerate(self.convs):
            out_temp = out_1[:, idx, :, :, :] # (N, C//self.split, H, W)
            out_1[:, idx, :, :, :] = conv(out_temp) # (N, sef.split, C//self.split, H, W)
        
        # step 2: weight
        out_2 = []
        for idx, conv_block in enumerate(self.conv_blocks):
            out_temp = out_1[:, idx, :, :, :] # (N, C//self.split, H, W)
            out_temp = conv_block(out_temp) # (N, C//self.split, H, W)
            out_2.append(out_temp) # a list type
        
        weight = torch.stack(out_2, dim=1) # (N, self.split, C//self.split, H, W)
        weight = weight.expand_as(out_1)

        # step3: softmax
        out_softmax =  self.softmax(weight)

        out = out_1 * out_softmax
        out = out.view(N, -1, H, W)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 11
'''not completed
class EMSAtt(nn.Module):
    r"""
    "ResT: An Efficient Transformer for Visual Recognition"
    """
    def __init__(self, channels, dim_key, dim_value, num_heads, H=7, W=7, ratio=3, transform=True):
        super(EMSAtt, self).__init__()
        self.dropout = 0.1
        self.H = H
        self.W = W
        self.fc_q = nn.Linear(channels, num_heads * dim_key)
        self.fc_k = nn.Linear(channels, num_heads * dim_key)

        self.fc_v = nn.Linear(channels, num_heads * dim_value)
        self.fc_out = nn.Linear(num_heads * dim_value, channels)
        self.dropout = nn.Dropout(self.dropout)

        self.ratio = ratio
        self.transform = transform

        if self.ratio > 1:
            self.

'''

# 12
class ShuffleAtt(nn.Module):
    r"""
    "SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS"
    """
    def __init__(self, channels, reduction=16, group=8):
        super(ShuffleAtt, self).__init__()
        self.group = group
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.group_norm = nn.GroupNorm(channels // (2 * group), channels // (2 * group))

        self.weight_1 = Parameter(torch.zeros(1, channels // (2 * group), 1, 1))
        self.bias_1 = Parameter(torch.ones(1, channels // (2 * group), 1, 1))

        self.weight_2 = Parameter(torch.zeros(1, channels // (2 * group), 1, 1))
        self.bias_2 = Parameter(torch.ones(1, channels // (2 * group), 1, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        N, C, H, W = input.shape

        # group into subfeatures
        out = input.view(N * self.group, -1, H, W) # (N * group, C // group, H, W)

        # channel_split
        out_0, out_1 = out.chunk(2, dim=1) # (N * group, C // (2 * group), H, W)

        # channel attention
        out_channel = self.avg_pool(out_0) # (N * group, C // (2 * group), 1, 1)
        out_channel = self.weight_1 * out_channel + self.bias_1 # (N * group, C // (2 * group), 1, 1)
        out_channel = self.sigmoid(out_channel) * out_0 # (N * group, C // (2 * group), H, W)

        # spatial attention
        out_spatial = self.group_norm(out_1) # (N * group, C // (2 * group), H, W)
        out_spatial = self.weight_2 * out_spatial + self.bias_2 # (N * group, C // (2 * group), H, W)
        out_spatial = self.sigmoid(out_spatial) * out_1 # (N * group, C // (2 * group), H, W)

        # concatenate along channels axis
        out = torch.cat([out_channel, out_spatial], dim=1) # (N * group, C // group, H, W)
        out = out.contiguous().view(N, -1, H, W)

        # channels shuffle
        out = self.channel_shuffle(out, 2)
        return out


    @staticmethod
    def channel_shuffle(input, groups):
        N, C, H, W = input.shape
        out = input.reshape(N, groups, -1, H, W)  # (N, groups, C // groups, H, W)
        out = out.permute(0, 2, 1, 3, 4) # (N, C // groups, groups, H, W)

        out = out.reshape(N, -1, H, W) # by gzb: what?
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 13
class Depth_Pointwise_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        if (kernel_size == 1):
            self.conv_depth = nn.Identity()
        else:
            self.conv_depth = nn.Conv1d(
                in_channels, in_channels,
                kernel_size=kernel_size,
                groups=in_channels,
                padding=kernel_size//2
            )
        
        self.conv_pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            groups=1
        )
    
    def forward(self, input):
        out = self.conv_depth(input)
        out = self.conv_pointwise(out)
        return out

class MuseAtt(nn.Module):
    r"""
    "MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning"
    """
    def __init__(self, channels, dim_key, dim_value, num_heads, dropout=.1):
        super(MuseAtt, self).__init__()
        self.channels = channels
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.fc_q = nn.Linear(channels, num_heads * dim_key)
        self.fc_k = nn.Linear(channels, num_heads * dim_key)
        self.fc_v = nn.Linear(channels, num_heads * dim_value)
        self.fc_out = nn.Linear(num_heads * dim_value, channels)

        self.conv1 = Depth_Pointwise_Conv1d(num_heads * dim_value, self.channels, kernel_size=1)
        self.conv3 = Depth_Pointwise_Conv1d(num_heads * dim_value, self.channels, kernel_size=3)
        self.conv5 = Depth_Pointwise_Conv1d(num_heads * dim_value, self.channels, kernel_size=5)

        self.dy_paras = nn.Parameter(torch.ones(3))

        self.softmax = nn.Softmax(-1)


        self.init_weights()

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        shape:
            input: (N, seq_length, input_size) # input_size == channels
        '''
        N, nq, _ = queries.shape # not use queries[:2], which return value with int type
        _, nk, _ = keys.shape

        query = self.fc_q(queries)
        query = query.view(N, nq, self.num_heads, self.dim_key).permute(0, 2, 1, 3) # (N, num_heads, nq, dim_key)

        key = self.fc_k(keys)
        key = key.view(N, nk, self.num_heads, self.dim_key).permute(0, 2, 3, 1) # (N, num_heads, dim_key, nk)

        value = self.fc_v(values)
        value = value.view(N, nk, self.num_heads, self.dim_value).permute(0, 2, 1, 3) # (N, num_heads, nk, dim_value)

        att = torch.matmul(query, key) # (N, num_heads, nq, nk)
        att = att / np.sqrt(self.dim_key)

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, value) # (N, num_heads, nq, dim_value)
        out = out.permute(0, 2, 1, 3).contiguous() # (N, nq, num_heads, dim_value)
        out = out.view(N, nq, self.num_heads * self.dim_value) # (N, nq, num_heads * dim_value)

        out = self.fc_out(out)

        value_2 = value.permute(0, 1, 3, 2).contiguous() # (N, num_heads, dim_value, nk)
        value_2 = value_2.view(N, -1, nk) # (N, num_heads * dim_value, nk)

        #self.dy_paras = self.softmax(self.dy_paras)
        self.dy_paras = nn.Parameter(self.softmax(self.dy_paras)) # by gzb: do not split into two line

        # from num_heads * dim_value to self.channels
        out2 = self.dy_paras[0] * self.conv1(value_2) + self.dy_paras[1] * self.conv3(value_2) + self.dy_paras[2] * self.conv5(value_2)

        out2 = out2.permute(0, 2, 1).contiguous() # (N, nk, self.channels)

        output = out + out2
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 14
class SpatialGEAtt(nn.Module):
    r"""
    Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks
    ori name: SpatialGroupEnhance
    """
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = Parameter(torch.ones(1, groups, 1, 1))
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def forward(self, input):
        '''
        shape:
            input: (N, C, H, W)
            output: (N, C, H, W)
        '''
        N, C, H, W = input.shape
        out_0 = input.view(N * self.groups, -1, H, W) # (N * groups, C // groups, H, W)

        out_1 = self.avg_pool(out_0) * out_0 # (N * groups, C // groups, H, W)
        out_1 = torch.sum(out_1, dim=1, keepdim=True) # (N * groups, 1, H, W)
        out_1 = out_1.view(N * self.groups, -1) # (N * groups, H * W)

        out_2 = torch.mean(out_1, dim=1, keepdim=True) # (N * groups, 1)
        out_2 = out_1 - out_2
        std = out_2.std(dim=1, keepdim=True) + 1e-5
        out_2 = out_2 / std # (N * groups, H * W)
        out_2 = out_2.view(N, self.groups, H, W)

        out_3 = self.weight * out_2 + self.bias  # (N, groups, H, W)
        out_3 = out_3.view(N * self.groups, 1, H, W) # (N * groups, 1, H, W)
        
        output = self.sigmoid(out_3) * out_0 # (N * groups, C // groups, H, W)
        output = output.view(N, C, H, W)
        return output


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 15
class DuableAtt(nn.Module):
    r"""
    'A2-Nets: Double Attention Networks'
    """
    def __init__(self, in_channels, out_channels_m, out_channels_n, reconstruct = True):
        super(DuableAtt, self).__init__()
        self.in_channels = in_channels
        self.out_channels_m = out_channels_m
        self.out_channels_n = out_channels_n
        self.reconstruct = reconstruct

        self.conv_a = nn.Conv2d(in_channels, out_channels_m, 1)
        self.conv_map = nn.Conv2d(in_channels, out_channels_n, 1)
        self.conv_vec = nn.Conv2d(in_channels, out_channels_n, 1)

        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(out_channels_m, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(-1)

        self.init_weights()

    def forward(self, input):
        N, C, H, W = input.shape
        assert C==self.in_channels

        out_a = self.conv_a(input) # (N, c_m, H, W)
        out_map = self.conv_map(input) # (N, c_n, H, W)
        out_vec = self.conv_vec(input) # (N, c_n, H, W)

        temp_a= out_a.view(N, self.out_channels_m, -1) # (N, c_m, H * W)
        #att_maps = F.softmax(out_map.view(N, self.out_channels_n, -1), dim=0) # (N, c_n, H * W)
        att_maps = self.softmax(out_map.view(N, self.out_channels_n, -1)) # (N, c_n, H * W)
        att_maps = att_maps.permute(0, 2, 1).contiguous() # (N, H*W, c_n)
        #att_vectors = F.softmax(out_vec.view(N, self.out_channels_n, -1), dim=0) # (N, c_n, H * W)
        att_vectors = self.softmax(out_vec.view(N, self.out_channels_n, -1)) # (N, c_n, H * W)

        # step 1: feature gating
        global_descriptors = torch.bmm(temp_a, att_maps) # (N, c_m, c_n)

        # step 2: feature distribution
        out = torch.matmul(global_descriptors, att_vectors) # (N, c_m, H * W)

        out = out.view(N, self.out_channels_m, H, W) # (N, in_channels, H,  W)
        if self.reconstruct:
            out = self.conv_reconstruct(out) # (N, in_channels, H * W)

        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 16
class FreeTransAtt(nn.Module):
    r"""
    An Attention Free Transformer
    """
    def __init__(self, channels, seq_length=49, simple=False):
        super(FreeTransAtt, self).__init__()
        
        self.channels = channels
        self.seq_length = seq_length

        self.fc_q = nn.Linear(channels, channels)
        self.fc_k = nn.Linear(channels, channels)
        self.fc_v = nn.Linear(channels, channels)

        self.sigmoid = nn.Sigmoid()

        if simple:
            self.bias_position = torch.zeros((seq_length, seq_length))
        else:
            self.bias_position = Parameter(torch.ones((seq_length, seq_length)))
        
        self.init_weights()
    
    def forward(self, input):
        '''
        shape:
            input: (bs, seq_length, input_size)
            output: (bs, seq_length, input_size)
        '''
        bs, seq_length, input_size = input.shape
        assert input_size == self.channels

        q = self.fc_q(input) # (bs, seq_length, input_size)
        k = self.fc_k(input).view(1, bs, seq_length, self.channels) # (1, bs, seq_length, input_size)
        v = self.fc_v(input).view(1, bs, seq_length, self.channels) # (1, bs, seq_length, input_size)

        #numerator=torch.sum(torch.exp(k+self.bias_position.view(n,1,-1,1))*v,dim=2)
        numerator = self.bias_position.view(seq_length, 1, -1, 1) # (seq_length, 1, seq_length, 1)
        numerator = torch.exp(k + numerator) * v  # (seq_length, bs, seq_length, input_size) ?
        numerator = torch.sum(numerator, dim=2) # (seq_length, bs, input_size) ?

        #denominator=torch.sum(torch.exp(k+self.bias_position.view(seq_length,1,-1,1)), dim=2)
        denominator = self.bias_position.view(seq_length, 1, -1, 1)
        denominator = torch.exp(k + denominator)
        denominator = torch.sum(denominator, dim=2) # (seq_length, bs, input_size)

        out = numerator / denominator # (seq_length, bs, input_size)
        out = self.sigmoid(q) * out.permute(1, 0, 2) # (bs, seq_length, input_size)

        return out



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 17 not complete
class OutlookAtt(nn.Module):
    r"""
    'VOLO: Vision Outlooker for Visual Recognition'
    """
    def __init__(self, channels, num_heads=1, kernel_size=3, padding=1, stride=1, qkv_bias=False, att_drop=0.1):
        super(OutlookAtt, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.kernel_size = 1
        self.padding = padding
        self.stride = stride

        self.channels_head = channels // num_heads
        self.scale = self.channels_head ** (-0.5)

        self.fc = nn.Linear(channels, channels, bia=qkv_bias)
        self.fc_att = nn.Linear(channels, kernel_size**4 * num_heads)
        self.drop_att = nn.Dropout(att_drop)

        self.fc_proj = nn.Linear(channels, channels)
        self.drop_proj = nn.Dropout(att_drop)

        self.unflod = nn.Unfold(kernel_size, padding, stride=1) # conv manually

        self.pool = nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True)

    def forward(self, input):
        N, C, H, W = input.shape

        # map to new feature space
        H = math.ceil(H / self.stride)
        W = math.ceil(W / self.stride)
        vec = self.unflod(input).reshape(N, self.num_heads, self.channels_head, self.kernel_size * self.kernel_size, H*W)

        # attention map
        att = self.pool(input) # (N, C, H, W)

# 18
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, act_layer=nn.GELU, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.Dropout(dropout)
        )
    def forward(self, input):
        out = self.mlp(input)
        return out

class VisionPermuteMLP(nn.Module):
    r"""
    'Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition'
    """
    def __init__(self, channels, H=25, W=20, seg_dim=8, qkv_bias=False, dropout=0.1):
        super(VisionPermuteMLP, self).__init__()
        self.seg_dim = seg_dim
        self.channels = channels

        self.fc_c = nn.Linear(channels, channels, bias=qkv_bias)
        #self.fc_h = nn.Linear(channels, channels, bias=qkv_bias)
        #self.fc_w = nn.Linear(channels, channels, bias=qkv_bias)
        self.fc_h = nn.Linear(H, H, bias=qkv_bias)
        self.fc_w = nn.Linear(W, W, bias=qkv_bias)

        self.reweighting = MLP(channels, channels//4, channels*3)

        self.fc_out = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=0)

    def forward(self, input):
        '''
        hc: hidden_channels
        oc: out_channels
        '''
        N, C, H, W = input.shape
        assert C == self.channels

        # embed for C
        out_embed_c = input.permute(0, 2, 3, 1).contiguous() # (N, H, W, C)
        out_embed_c = self.fc_c(out_embed_c) # (N, H, W, C)
        out_embed_c = out_embed_c.permute(0, 3, 1, 2).contiguous()

        # param
        seg = C // self.seg_dim
        out_temp = input.view(N, self.seg_dim, seg, H, W) # (N, seg_dim, seg, H, W)

        # embed for H
        '''
        out_embed_h = out_temp.permute(0, 1, 4, 3, 2).contiguous().view(N, self.seg_dim, W, H*seg) # (N, seg_dim, W, H*seg)
        out_embed_h = self.fc_h(out_embed_h).view(N, self.seg_dim, W, H, seg).permute(0, 1, 4, 3, 2).contiguous() # (N, seg_dim, seg, H, W)
        out_embed_h = out_embed_h.view(N, C, H, W) # (N, C, H, W)
        
        # embed for w
        out_embed_w = out_temp.permute(0, 1, 3, 4, 2).contiguous().view(N, self.seg_dim, H, W*seg) # (N, seg_dim, H, W*seg)
        out_embed_w = self.fc_w(out_embed_w).view(N, self.seg_dim, H, W, seg).permute(0, 1, 4, 2, 3).contiguous() # (N, seg_dim, seg, H, W)
        out_embed_w = out_embed_w.view(N, C, H, W) # (N, C, H, W)
        '''

        out_embed_h = input.permute(0, 1, 3, 2).contiguous() # (N, C, W, H)
        out_embed_h = self.fc_h(out_embed_h).permute(0, 1, 3, 2) # (N, C, W, H)
        out_embed_w = self.fc_w(input) # (N, C, H, W)

        # fuse
        out = out_embed_c + out_embed_h + out_embed_w

        weight = torch.flatten(out, start_dim=2, end_dim=-1) # (N, C, H*W)
        weight = torch.mean(weight, dim=2) # (N, C)
        weight = self.reweighting(weight).view(N, C, 3).permute(2, 0, 1).contiguous() # (3, N, C)
        weight = self.softmax(weight).unsqueeze(-1).unsqueeze(-1) # (3, N, C, 1, 1)

        output = weight[0] * out_embed_c + weight[1] * out_embed_h + weight[2] * out_embed_w
        output = output.permute(0, 2, 3, 1).contiguous() # (N, H, W, C)
        output = self.fc_out(output).permute(0, 3, 1, 2) # (N, C, H, W)

        output = self.dropout(output)

        return output


# 19 not complete
#class 
'CoAtNet: Marrying Convolution and Attention for All Data Sizes'

# 20

# 21
class PolarizedSelfAtt(nn.Module):
    r"""
    'Polarized Self-Attention: Towards High-quality Pixel-wise Regression'
    """
    def __init__(self, channels=512, sequential=True):
        super().__init__()
        self.channels = channels
        self.sequential = sequential

        self.conv_channel_1 = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.conv_channel_2 = nn.Conv2d(channels, 1, kernel_size=1)
        self.conv_channel_3 = nn.Conv2d(channels // 2, channels, kernel_size=1)
        
        self.conv_spatial_1 = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.conv_spatial_2 = nn.Conv2d(channels, channels // 2, kernel_size=1)

        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)

        self.ln = nn.LayerNorm(channels)
        self.sigmoid = nn.Sigmoid()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        '''
        shape:
            input: (N, C, H, W)
            output: (N, C, H, W)
        '''
        N, C, H, W = input.shape
        assert C == self.channels

        # channel-only self-attention: as weight channel
        out_channel_1 = self.conv_channel_1(input).view(N, C//2, -1) # (N, C//2, H * W)

        out_channel_2 = self.conv_channel_2(input).view(N, -1, 1) # (N, 1, H, W) to (N, H * W, 1)
        out_channel_2 = self.softmax_channel(out_channel_2) # (N, H * W, 1)

        out_channel_3 = torch.matmul(out_channel_1, out_channel_2).unsqueeze(-1) # (N, C//2, 1, 1)
        out_channel_3 = self.conv_channel_3(out_channel_3).view(N, C, 1).permute(0, 2, 1).contiguous() # (N, 1, C)
        out_channel_3 = self.ln(out_channel_3)
        out_channel_3 = self.sigmoid(out_channel_3).permute(0, 2, 1).contiguous().view(N, C, 1, 1) # (N, C, 1, 1)

        out_channel = out_channel_3 * input # (N, C, H, W)

        # spatial-only self-attention: as weight_spa
        if self.sequential:
            input_spatial = out_channel
        else:
            input_spatial = input

        out_spatial_1 = self.conv_spatial_1(input_spatial).view(N, C//2, -1) # (N, C//2, H * W)

        out_spatial_2 = self.conv_spatial_2(input_spatial) # (N, C//2, H, W)
        out_spatial_2 = self.pool(out_spatial_2) # (N, C//2, 1, 1)
        out_spatial_2 = out_spatial_2.permute(0, 2, 3, 1).contiguous().view(N, 1, C//2) # (N, 1, C//2)
        out_spatial_2 = self.softmax_spatial(out_spatial_2) # (N, 1, C//2)

        out_spatial_3 = torch.matmul(out_spatial_2, out_spatial_1) # (N, 1, H*W)
        out_spatial_3 = self.sigmoid(out_spatial_3.view(N, 1, H, W)) # (N, 1, H, W)

        # fuse spa and channel
        if self.sequential:
            output = out_spatial_3 * out_channel
        else:
            out_spa = out_spatial_3 * input
            output = out_spa + out_channel

        return output     

# 22
class CoTAtt(nn.Module):
    r"""
    'Contextual Transformer Networks for Visual Recognition---arXiv 2021.07.26'
    """
    def __init__(self, channels=512, kernel_size=3):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size

        self.embed_key = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.embed_value = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        reduction = 4
        self.embed_att = nn.Sequential(
            nn.Conv2d(2*channels, 2*channels // reduction, 1, bias=False),
            nn.BatchNorm2d(2 * channels // reduction),
            nn.ReLU(),
            nn.Conv2d(2*channels// reduction, kernel_size * kernel_size * channels, 1)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        '''
        shape:
            input: (N, C, H, W)
            output: (N, C, H, W)
        '''
        N, C, H, W = input.shape
        assert C == self.channels
        out_key = self.embed_key(input) # (N, C, H, W)
        out_value = self.embed_value(input).view(N, C, -1) # (N, C, H*W)

        out = torch.cat([out_key, input], dim=1) # (N, C*2, H, W)

        out_att = self.embed_att(out) # (N, C*k*k, H, W) k: kernel_size
        out_att = out_att.view(N, C, -1, H, W) # (N, C, kernel_size * kernel_size, H, W)
        out_att = torch.mean(out_att, dim=2, keepdim=False) # (N, C, H, W)
        out_att = out_att.view(N, C, -1)

        output = self.softmax(out_att) * out_value
        output = output.view(N, C, H, W)

        return out_key + output # (N, C, H, W)


# 23 not completed? 
class ResidualAtt(nn.Module):
    r"""
    'Residual Attention: A Simple but Effective Method for Multi-Label Recognition---ICCV2021'
    """
    def __init__(self, channels=512 , num_classes=1000, factor=0.2):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv2d(channels, num_classes, kernel_size=1, bias=False)

    def forward(self, input):
        '''
        shape:
            input: (N, C, H, W)
            output: (N, numclasses)
        '''
        out = self.conv(input) # (N, num_classes, H, W)
        out = torch.flatten(out, start_dim=2, end_dim=-1) # (N, num_classes, H*W)
        out_avg = torch.mean(out, dim=2) # (N, num_classes)
        out_max = torch.max(out, dim=2)[0] # (N, num_classes)

        output = out_avg + self.factor * out_max
        
        return output

# 24
class SplitAtt(nn.Module):
    def __init__(self, channels=512, kernel_size=3):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.GELU(),
            nn.Linear(channels, channels*kernel_size, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)


    def forward(self, input):
        N, K, H, W, C = input.shape
        assert C == self.channels
        assert K == self.kernel_size

        # reshape
        input = input.view(N, K, -1, C) # (N, K, H*W, C)
        
        # out_1: a_hat
        out_1 = torch.sum(input, 1) # (N, H*W, C)
        out_1 = torch.sum(out_1, 1) # (N, C)
        out_1 = self.mlp(out_1) # (N, C*K)
        out_1 = out_1.view(N, K, C)

        # out_2: a_bar
        out_2 = self.softmax(out_1) # (N, K, C)

        # attention
        att = out_2.unsqueeze(-2) # (N, K, 1, C)

        # output
        output = att * input # (N, K, H*W, C)

        output = torch.sum(output, dim=1).view(N, H, W, C)

        return output

class S2Att(nn.Module):
    r"""
    'S²-MLPv2: Improved Spatial-Shift MLP Architecture for Vision---arXiv 2021.08.02'
    """
    def __init__(self, channels=512 ):
        super().__init__()
        self.channels = channels

        self.fc_1 = nn.Linear(channels, channels*3)
        self.fc_2 = nn.Linear(channels, channels)
        self.split_att = SplitAtt(channels)

    def forward(self, input):
        '''
        shape:
            input: (N, C, H, W)
            output: (N, C, H, W)
        '''
        N, C, H, W = input.shape
        assert C == self.channels

        # reshape
        input = input.permute(0, 2, 3, 1).contiguous() # (N, H, W, C)

        # step 1
        out_1 = self.fc_1(input) # (N, H, W, C*3)

        # step 2: shift
        out_shift_1 = self.spatial_shift1(out_1[:, :, :, :C]) # (N, H, W, C)
        out_shift_2 = self.spatial_shift2(out_1[:, :, :, C:C*2]) # (N, H, W, C)
        out_shift_3 = out_1[:, :, :, C*2:]

        # step 3: stack
        out_2 = torch.stack([out_shift_1, out_shift_2, out_shift_3], dim=1) # (N, K, H, W, C)

        # step 4: att
        out_3 = self.split_att(out_2) # (N, H, W, C)
        out_3 = self.fc_2(out_3) # (N, H, W, C)

        output = out_3.permute(0, 3, 1, 2).contiguous() # (N, C, H, W)

        return output


    def spatial_shift1(self, input):
        N, H, W, C = input.shape
        input[:, 1:, :, :C//4] = input[:, :H-1, :, :C//4]
        input[:, :H-1, :, C//4:C//2] = input[:, 1:, :, C//4:C//2]
        input[:, :, 1:, C//2:C*3//4] = input[:, :, :W-1, C//2:C*3//4]
        input[:, :, :W-1, C*3//4:] = input[:, :, 1:, C*3//4:]
        return input

    def spatial_shift2(self, input):
        N, H, W, C = input.shape
        input[:, :, 1:, :C//4] = input[:, :, :W-1, :C//4]
        input[:, :, :W-1, C//4:C//2] = input[:, :, 1:, C//4:C//2]
        input[: , 1:, :, C//2:C*3//4] = input[:, :H-1,:, C//2:C*3//4]
        input[:, :H-1, :, 3*C//4:] = input[:, 1:, :, 3*C//4:]
        return input

# 25

'Global Filter Networks for Image Classification---arXiv 2021.07.01'

# 26
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, input):
        out= self.conv(input)
        if self.bn is not None:
            out = self.bn(out)
        if self.relu is not None:
            out = self.relu(out)
        return out

class AttGate(nn.Module):
    def __init__(self):
        super(AttGate, self).__init__()
        kernel_size = 7
        #self.conv = BasicConv(2, 1, kernel_size, padding=kernel_size//2, relu=False)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
        )

    def forward(self, input):
        '''
        shape:
            input: (N, C, H, W)
        '''
        # step 1:
        out_max = torch.max(input, dim=1, keepdim=True)[0] # (N, 1, H, W)
        out_avg = torch.mean(input, dim=1, keepdim=True) # (N, 1, H, W)
        out_1 = torch.cat([out_max, out_avg], dim=1) #(N, 2, H, W)

        out_2 = self.conv(out_1) # (N, 1, H, W)
        scale = torch.sigmoid_(out_2)

        output = input * scale # (N, C, H, W)
        return output

class TripleAtt(nn.Module):
    r"""
    'Rotate to Attend: Convolutional Triplet Attention Module---CVPR 2021'
    """
    def __init__(self, spatial=True):
        super(TripleAtt, self).__init__()
        self.spatial = spatial

        self.att_gate_1 = AttGate()
        self.att_gate_2 = AttGate()

        if spatial:
            self.att_gate_3 = AttGate()

    def forward(self, input):
        '''
        shape:
            input: (N, C, H, W)

        '''
        input_1 = input.permute(0, 2, 1, 3).contiguous() # (N, H, C, W)
        input_2 = input.permute(0, 3, 2, 1).contiguous() # (N, W, H, C)

        out_1 = self.att_gate_1(input_1) # (N, H, C, W); perform conv for H
        out_1 = out_1.permute(0, 2, 1, 3).contiguous() # (N, C, H, W)

        out_2 = self.att_gate_2(input_2) # (N, W, H, C); perform conv for W
        out_2 = out_2.permute(0, 3, 2, 1) # (N, C, H, W)

        if self.spatial:
            out_3 = self.att_gate_3(input) # (N, C, H, W); perform conv for C
            output = 1/3 * (out_3 + out_1 + out_2)
        else:
            output = 1/2 * (out_1 + out_2)

        return output

# 27
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    
    def forward(self, input):
        out = self.relu(input + 3) / 6
        out = input * out
        return out

class CoordAtt(nn.Module):
    r"""
    'Coordinate Attention for Efficient Mobile Network Design---CVPR 2021
    """
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAtt, self).__init__()
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        inter_channels = max(8, in_channels//reduction)
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            h_swish()
        )

        self.conv_h = nn.Conv2d(inter_channels, out_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(inter_channels, out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        N, C, H, W = input.shape

        residual = input

        # step 1: pool
        out_h = self.pool_h(input) # (N, C, H, 1)
        out_w = self.pool_w(input) # (N, C, 1, W)
        out_w = out_w.permute(0, 1, 3, 2).contiguous() # (N, C, W, 1)

        # step 2: joint conv
        out_hw = torch.cat([out_h, out_w], dim=2) # (N, C, H+W, 1)
        out_hw = self.conv_1(out_hw) # (N, C//reduction or 8, H+W, 1)

        out_h, out_w = torch.split(out_hw, [H, W], dim=2) # (N, C, H, 1) and (N, C, W, 1)
        out_w = out_w.permute(0, 1, 3, 2).contiguous() # (N, C, 1, W)

        # step 3: conv respectively
        out_h = self.conv_h(out_h) # (N, C, H, 1)
        out_w = self.conv_w(out_w) # # (N, C, 1, W)

        # step 4: compute scale factor
        scale_h = self.sigmoid(out_h)
        scale_w = self.sigmoid(out_w)

        # step 5: fuse
        output = residual * scale_h * scale_w # (N, C, H, W)

        return output

# 28 not completed
'MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer---ArXiv 2021.10.05'
class Attention(nn.Module):
    def __init__(self, channels, num_heads, head_channels, dropout):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = head_channels

        self.scale = head_channels ** (-0.5)

        inter_channels = num_heads * head_channels

        self.fc_qkv = nn.Linear(channels, inter_channels * 3, bias=False)
        
        if inter_channels == channels: # inter_channels == channels
            self.fc_out = nn.Identity()
        else:
            self.fc_out = nn.Sequential(
                nn.Linear(inter_channels, channels),
                nn.Dropout(dropout)
            )
        
        self.softmax =nn.Softmax(dim=-1)

    def forward(self, input):
        qkv = self.fc_qkv(input).chunk(3, dim=-1)

        query, key, value = 0

# 29
class ParNetAtt(nn.Module):
    r"""
    'Non-deep Networks---ArXiv 2021.10.20'
    """
    def __init__(self, channels=512):
        super().__init__()
        self.channels = channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels)
        )

        self.covn2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.silu = nn.SiLU()

    def forward(self, input):
        '''
        shape:
            input: (N, C, H, W)
            output: (N, C, H, W)
        '''
        N, C, H, W = input.shape
        assert C == self.channels

        out_1 = self.conv1(input)
        out_2 = self.covn2(input)

        out_3 = self.conv3(input) # weight
        out_3 = out_3 * input

        output = self.silu(out_1 + out_2 + out_3)

        return output

# 30
class UFOAtt(nn.Module):
    r"""
    'UFO-ViT: High Performance Linear Vision Transformer without Softmax---ArXiv 2021.09.29'
    Scaled dot-product attention
    """
    def __init__(self, channels, dim_key, dim_value, num_heads, dropout=.1):
        '''
        param:
            channels: Output dimensionality of the model
            dim_key: Dimensionality of queries and keys
            dim_value: Dimensionality of values
            num_heads: Number of heads
        '''
        super(UFOAttention, self).__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.fc_q = nn.Linear(channels, num_heads * dim_key)
        self.fc_k = nn.Linear(channels, num_heads * dim_key)
        self.fc_v = nn.Linear(channels, num_heads * dim_value)

        self.fc_out = nn.Linear(num_heads * dim_value, channels)

        self.dropout = nn.Dropout(dropout)
        self.gamma = Parameter(torch.randn((1, num_heads, 1, 1)))

    
        self.init_weights()

    def forward(self, queries, keys, values):
        bs, nq, _ = queries.shape
        _, nk, _ = keys.shape

        query = self.fc_q(queries) # (bs, nq, nh * dk)
        query = query.view(bs, nq, self.num_heads, self.dim_key).permute(0, 2, 1, 3) # (bs, nh, nq, dk)

        key = self.fc_k(keys) # (bs, nk, nh * dk)
        key = key.view(bs, nk, self.num_heads, self.dim_key).permute(0, 2, 3, 1) # (bs, nh, dk, nk)

        value = self.fc_v(values) # (bs, nk, nh * dv)
        value = value.view(bs, nk, self.num_heads, self.dim_value).permute(0, 2, 1, 3) # (bs, nh, nk, dv)

        out = torch.matmul(key, value) # (bs, nh, dk, dv)
        out_norm_1 = self.Xnorm(out, self.gamma) # (bs, nh, dk, dv)
        out_norm_2 = self.Xnorm(query, self.gamma) # (bs, nh, nq, dk)

        output = torch.matmul(out_norm_2, out_norm_1) # (bs, nh, nq, dv)
        output = output.permute(0, 2, 1, 3).contiguous() # (bs, nq, nh, dv)
        output = output.view(bs, nq, -1) # (bs,nq, nh * dv)
        output = self.fc_out(output)

        return output

    def Xnorm(self, input, gamma):
        out = torch.norm(input, p=2, dim=-1, keepdim=True)
        out = input * gamma / out
        return out


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

## Conv
class Involution(nn.Module):
    def __init__(self, in_channel=4, kernel_size=1, stride=1, group=1, ratio=4):
        super().__init__()
        self.in_channels = in_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.group = group
        assert in_channel % group == 0
        self.group_channels = in_channel // group

        self.avg_pool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//ratio, kernel_size=1),
            nn.BatchNorm2d(in_channel // ratio),
            nn.ReLU(),
            nn.Conv2d(in_channel // ratio, kernel_size * kernel_size * group, kernel_size=1)
        )

        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=kernel_size//2)

    def forward(self, input):
        N, C, H, W = input.shape

        # step 1
        out_1 = self.avg_pool(input) # (N, C, H//stride, W//stide)
        out_1 = self.conv(out_1) # (N, group*k*k, H//stride, W//stride)

        # step 2
        N, C1, H1, W1 = out_1.shape
        out_2 = out_1.view(N, self.group, -1, H1, W1) # (N, group, k*k, H//stride, W//stride)
        out_2 = out_2.unsqueeze(2) # (N, group, 1, k*k, H//stride, W//stride)

        # step 3
        out_3 = self.unfold(input)
        out_3 = out_3.view(N, self.group, C//self.group, self.kernel_size*self.kernel_size, H//self.stride, W//self.stride)

        # step 4
        output = (out_3 * out_2).sum(dim=3) # (N, group, C//group, H//stride, W//stride)
        output = output.view(N, C, H//self.stride, W//self.stride)

        return output



def debug_SELayer():
    x = torch.randn(size = (4,8,20,20))        
    selayer = SELayer(channel = 8, reduction = 2)
    out = selayer(x)    
    print(out.size()) 

    '''
    output: 
    torch.Size([4, 8, 20, 20])
    '''   

def debug_CBAtt_Res():
    x = torch.randn(size = (4,8,20,20))  
    cba = CBAtt_Res(8,16,reduction = 2,stride = 1) 
    y = cba(x)
    print('y.size:',y.size())   

    '''
    y.size: torch.Size([4, 16, 20, 20])
    '''

def debug_SKEConv():
    x = torch.randn(size = (4,8,20,20))  
    ske = SKEConv(8,stride = 2)
    y = ske(x)
    print('y.size:',y.size())   

    '''
    y.size: torch.Size([4, 16, 10, 10])
    '''

def debug_SelfAtt_Spatial():
    x = torch.randn(size = (4,16,20,20))  
    self_atten_spatial = SelfAtt_Spatial(16,4)
    y = self_atten_spatial(x)
    print('y.size:',y[0].size())   

    '''
    y.size: torch.Size([4, 16, 20, 20])
    '''

def debug_SelfAtt_channel():
    x = torch.randn(size = (4,8,20,20))  
    self_atten_channel = SelfAtt_channel(8, 8)
    y = self_atten_channel(x)
    print('y.size:',y[0].size()) 

    '''
    output:
    y.size: torch.Size([4, 8, 20, 20])
    '''

def debug_NonLocalBlockND():
    x = torch.randn(size = (4,16,20,20))  
    non_local = NonLocalBlockND(16,inter_channels = 8,dimension = 2)
    y = non_local(x)
    print('y.size:',y.size())

    '''
    output:
    y.size: torch.Size([4, 16, 20, 20])
    '''

def debug_ExternalAtt():
    input = torch.randn(50, 49, 512)
    extAtt_layer = ExternalAtt(in_channles=512, inter_channels=8)
    output = extAtt_layer(input)
    print (output.shape)

    '''
    output:
    y.size: torch.Size([50, 49, 512])
    '''

def debug_ScaledDotProductAtt():
    input=torch.randn(50,49,512)
    sa = SDPAtt(channels=512, dim_key=512, dim_value=512, num_heads=8)
    output=sa(input,input,input)
    print(output.shape)

def debug_SimSDPAtt():
    input=torch.randn(50,49,512)
    ssa = SimSDPAtt(channels=512, num_heads=8)
    output=ssa(input,input,input)
    print(output.shape)

def debug_SKAtt():
    input=torch.randn(50,512,7,7)
    se = SKAtt(channels=512,reduction=8)
    output=se(input)
    print(output.shape)

def debug_BAMBlock():
    input=torch.randn(50,512,7,7)
    bam = BAMBlock(channels=512,reduction=16,dia_val=2)
    output=bam(input)
    print(output.shape)

def dubug_ECAtt():
    input=torch.randn(50,512,7,7)
    att_eca = ECAtt(kernel_size=3)
    output=att_eca(input)
    print(output.shape)

def debug_DualAtt():
    input=torch.randn(50,512,7,7)
    danet=DualAtt(channels=512,kernel_size=3,H=7,W=7)
    print(danet(input).shape)

def debug_PyramidSplitAtt():
    input=torch.randn(50,512,7,7)
    att_psa = PyramidSplitAtt(channels=512, reduction=8)
    output=att_psa(input)
    #a=output.view(-1).sum()
    #a.backward()
    print(output.shape)

def debug_ShuffleAtt():
    input=torch.randn(50,512,7,7)
    att = ShuffleAtt(channels=512,group=8)
    output=att(input)
    print(output.shape)

def debug_MuseAtt():
    input=torch.randn(50,49,512)
    sa = MuseAtt(channels=512, dim_key=512, dim_value=512, num_heads=8)
    output=sa(input,input,input)
    print(output.shape)

def debug_SGEAtt():
    input=torch.randn(50,512,7,7)
    sge = SpatialGEAtt(groups=8)
    output=sge(input)
    print(output.shape)

def debug_DuableAtt():
    input=torch.randn(50,512,7,7)
    a2 = DuableAtt(512,128,128,True)
    output=a2(input)
    print(output.shape)

def debug_FreeTransAtt():
    input=torch.randn(50,49,512)
    aft_full = FreeTransAtt(channels=512, seq_length=49)
    output=aft_full(input)
    print(output.shape)    

def debug_VisionPermuteMLP():
    input=torch.randn(64, 512, 8, 8)
    seg_dim=8
    vip=VisionPermuteMLP(512, seg_dim)
    out=vip(input)
    print(out.shape)

def debug_PolarizedSelfAtt():
    input=torch.randn(1,512,7,7)
    psa = PolarizedSelfAtt(channels=512, sequential=False)
    output=psa(input)
    print(output.shape)

def debug_CoTAtt():
    input=torch.randn(50,512,7,7)
    cot = CoTAtt(channels=512, kernel_size=3)
    output=cot(input)
    print(output.shape)

def debug_ResidualAtt():
    ResidualAtt
    input=torch.randn(50,512,7,7)
    resatt = ResidualAtt(channels=512, num_classes=1000, factor=0.2)
    output=resatt(input)
    print(output.shape)

def debug_S2Att():
    input=torch.randn(50,512,7,7)
    s2att = S2Att(channels=512)
    output=s2att(input)
    print(output.shape)

def debug_TripleAtt():
    input=torch.randn(50,512,7,7)
    triplet = TripleAtt()
    output=triplet(input)
    print(output.shape)

def debug_CoordAtt():
    input = torch.randn(50, 512, 7, 7)
    att_conv = CoordAtt(512, 512, reduction=16)
    output = att_conv(input)
    print (output.shape)

def debug_ParNetAtt():
    input=torch.randn(50,512,7,7)
    pna = ParNetAtt(channels=512)
    output=pna(input)
    print(output.shape)

def debug_UFOAtt():
    input=torch.randn(50,49,512)
    ufo = UFOAtt(channels=512, dim_key=512, dim_value=512, num_heads=8)
    output=ufo(input,input,input)
    print(output.shape)

def debug_conv_Involution():
    input=torch.randn(1,4,64,64)
    involution=Involution(kernel_size=3,in_channel=4,stride=2)
    out=involution(input)
    print(out.shape)

if __name__ == '__main__':
    print ("utils_attention.py")
    debug_conv_Involution()