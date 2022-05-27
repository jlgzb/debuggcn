from torch import nn

# VTE-STE
from models.block.vanilla_transformer_encoder import Transformer as VTE
from models.block.strided_transformer_encoder import Transformer as STE

# 20220129
class VTESTE(nn.modules):
    def __init__(self, args):
        super().__init__()

        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        stride_num = args.stride_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        self.encoder = nn.Sequential(
            nn.Conv1d(2*self.num_joints_in, channel, kernel_size=1),
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

        self.Transformer = VTE(layers, channel, d_hid, length=length)
        self.Transformer_reduce = STE(len(stride_num), channel, d_hid, \
            length=length, stride_num=stride_num)
        
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3*self.num_joints_out, kernel_size=1)
        )

        self.fcn_1 = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3*self.num_joints_out, kernel_size=1)
        )

    def forward(self, x):
        ''' by gzb: ori
        shape:
            input: (N, C, T, V) == (N, 2, T, 17)
            output: (N, 3, T, 17)

        for cigcn:
            input: (N, C, V, T)

        shape for VTE:
            input: (N, T, channel)
            output: (N, T, channel)

        shape for STE:
            input: (N, T, channel)
            output: (N, 1, channel)
        '''
        
        x = x.permute(0, 3, 2, 1).contiguous() # (N, T, V, C)
        x_shape = x.shape
        x = x.view(x_shape[0], x_shape[1], -1) # (N, T, V*C)

        x = self.Transformer(x) # (N, T, channel)

        x_VTE = x
        x_VTE = x_VTE.permute(0, 2, 1).contiguous() # (N, channel, T)
        x_VTE = self.fcn_1(x_VTE) # (N, V*C, T)

        x_VTE = x_VTE.view(x_shape[0], self.num_joints_out, -1, x_VTE.shape[2]) # (N, V, C, T)
        #x_VTE = x_VTE.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # (N, C=3, T, V) -> (N, C=3, T, V, 1)

        x = self.Transformer_reduce(x) # (N, 1, channel)
        x = x.permute(0, 2, 1).contiguous() # (N, channel, 1)
        x = self.fcn(x) # (N, V*C, 1)

        x = x.view(x_shape[0], self.num_joints_out, -1, x.shape[2]) # (N, V, 3, 1)
        #x = x.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # (N, 3, 1, V) -> (N, C=3, 1, V, 1)
        
        return x, x_VTE # (N, C=3, 1, V, 1); (N, C=3, T, V, 1)