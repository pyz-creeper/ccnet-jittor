import jittor as jt
from jittor import nn
from .util_module import ConvModule, Dropout2d
from .cc_attention import CrissCrossAttention


class CCHead(nn.Module):
    def __init__(self, in_index=3, recurrence=2, in_channels=2048, channels=512, dropout_rate=0.1, num_classes=150):
        self.in_index = in_index
        self.channels = channels
        self.recurrence = recurrence
        self.cca = CrissCrossAttention(self.channels)

        # default convs num is 2
        self.convs = [
            ConvModule(in_channel=in_channels,out_channel=channels,kernel_size=3,padding=1,dilation=1),
            ConvModule(in_channel=channels,out_channel=channels,kernel_size=3,padding=1,dilation=1),
            ]
        self.conv_cat = ConvModule(in_channel=in_channels+channels,out_channel=channels,kernel_size=3,padding=1,dilation=1)

        # used for cls_seg
        self.dropout = Dropout2d(p=dropout_rate)
        self.conv_seg = nn.Conv(channels,num_classes,kernel_size=1)


    def cls_seg(self, feat):
        feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def execute(self, inputs):
        '''
        implementation of mmseg use only the layer4 output
        we do the same
        '''
        x = inputs[self.in_index]
        output = self.convs[0](x)
        for _ in range(self.recurrence):
            output = self.cca(output)
        output = self.convs[1](output)
        output = self.conv_cat(jt.concat([x, output], dim=1))
        output = self.cls_seg(output)
        return output