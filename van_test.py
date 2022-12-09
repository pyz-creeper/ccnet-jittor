# used to test loading pretrained VAN
from networks.ccnet import VAN_CCnet
import jittor as jt

if __name__ == "__main__":
    net = VAN_CCnet('vanilla',2)
    input = jt.rand(1,3,512,683)
    outs = net(input)
    for out in outs:
        print(out.shape)