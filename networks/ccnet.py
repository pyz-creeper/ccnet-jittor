import jittor as jt
from jittor import nn
from .util_module import ConvModule, Dropout2d
from .cc_attention import CrissCrossAttention, DilatedCrissCrossAttention, NeighborhoodCrissCrossAttention
from .resnet import Resnet101
from .van import van_large,van_base


class CCHead(nn.Module):
    def __init__(self, in_index=-1, recurrence=2, attention_block="vanilla", in_channels=2048, channels=512, dropout_rate=0.1, num_classes=150): # 150 classes and a background
        self.in_index = in_index
        self.channels = channels
        self.recurrence = recurrence
        if attention_block == "dilated":
            self.cca = DilatedCrissCrossAttention(self.channels, dilated=2)
        elif attention_block == "neighborhood":
            self.cca = NeighborhoodCrissCrossAttention(self.channels)
        else:
            self.cca = CrissCrossAttention(self.channels)

        # default convs num is 2
        self.convs = nn.ModuleList(
            ConvModule(in_channel=in_channels,out_channel=channels,kernel_size=3,padding=1,dilation=1),
            ConvModule(in_channel=channels,out_channel=channels,kernel_size=3,padding=1,dilation=1),
        )
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


class AuxiliaryAttentionHead(nn.Module):
    def __init__(self, in_index=-2, in_channels=1024, channels=256, dropout_rate=0.1, num_classes=150):
        self.in_index = in_index
        self.channels = channels
        self.convs = nn.ModuleList(ConvModule(in_channel=in_channels,out_channel=channels,kernel_size=3,padding=1,dilation=1))
        self.dropout = Dropout2d(p=dropout_rate)
        self.conv_seg = nn.Conv(channels,num_classes,kernel_size=1)
    
    def execute(self, inputs):
        x = inputs[self.in_index]
        feat = self.convs(x)
        feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output


class CCnet(nn.Module):
    def __init__(self, attention_block, recurrence, pretrained) -> None:
        self.backbone = Resnet101(pretrained=pretrained)
        self.decode_head = CCHead(attention_block=attention_block, recurrence=recurrence)
        self.auxiliary_head = AuxiliaryAttentionHead()

    def execute(self,x):
        output_features = self.backbone(x)
        output_main = self.decode_head(output_features)
        output_main = nn.resize(output_main,x.shape[2:],mode="bilinear")
        output_aux = self.auxiliary_head(output_features)
        output_aux = nn.resize(output_aux,x.shape[2:],mode="bilinear")
        return output_main,output_aux
    
    def test_encode_aux(self, input):
        input = jt.array(input)
        input = [input, input,input[:,:1024,:,:],input]
        output_main = self.decode_head(input)
        output_aux = self.auxiliary_head(input)
        return output_main, output_aux
    
    def test_all_forward(self, input):
        input = jt.array(input)
        input = self.backbone(input)
        out_decode = self.decode_head(input)
        out_aux = self.auxiliary_head(input)
        return out_decode, out_aux       
        

class VAN_CCnet(nn.Module):
    def __init__(self, attention_block, recurrence,pretrained) -> None:
        self.backbone = van_base(pretrained=pretrained)
        self.decoder = CCHead(in_channels=512,attention_block=attention_block, recurrence=recurrence)
        self.aux_decoder = AuxiliaryAttentionHead(in_index=-1,in_channels=512)

    def execute(self,x):
        output_features = self.backbone(x)
        output_main = self.decoder(output_features)
        output_main = nn.resize(output_main,x.shape[2:],mode="bilinear")
        output_aux = self.aux_decoder(output_features)
        output_aux = nn.resize(output_aux,x.shape[2:],mode="bilinear")
        return output_main,output_aux