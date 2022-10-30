import jittor as jt
from jittor import nn

class ConvModule(nn.Module):
    '''
    conv - batchnorm - relu
    '''
    def __init__(self,in_channel,out_channel,kernel_size,padding,dilation) -> None:
        self.conv = nn.Conv(in_channel,out_channel,kernel_size,padding=padding,dilation=dilation)
        self.norm = nn.BatchNorm(out_channel)
        self.activate = nn.Relu()

    def execute(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activate(x)
        return x
    

class Dropout2d(nn.Module):
    '''
    implementation of dropout 2D
    reference: pytorch
    '''
    def __init__(self, p = 0.5, is_train = False) -> None:
        self.p = p

    def execute(self,x):
        if self.is_training():
            mask = (jt.rand([1,x.shape[1],1,1]) > self.p).float()
            return x * mask
        else:
            return x
