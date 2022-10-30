import jittor as jt
from jittor import nn
from cc_attention import CrissCrossAttention


class CCHead(nn.Module):
    def __init__(self, in_index=3, recurrence=2, in_channels=2048, channels=512, **kwargs):
        super(CCHead, self).__init__(num_convs=2, **kwargs)
        self.in_index = in_index
        self.channels = channels
        self.recurrence = recurrence
        self.cca = CrissCrossAttention(self.channels)
        # add convs default convs num is 2
        self.convs = []

    def cls_seg(self, feat):
        pass

    def forward(self, inputs):
        '''
        implementation of mmseg use only the layer4 output
        we do the same
        '''
        x = inputs[self.in_index]
        output = self.convs[0](x)
        for _ in range(self.recurrence):
            output = self.cca(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(jt.concat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
