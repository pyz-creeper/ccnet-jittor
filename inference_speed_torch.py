import torch
torch.cuda.set_device("cuda:2")
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import time
import numpy as np

def get_mmseg_model():
    config_file = '../gpu8/mmsegmentation/configs/ccnet/ccnet_r101-d8_512x512_80k_ade20k.py'
    checkpoint_file = '../gpu8/mmsegmentation/checkpoints/ccnet_r101-d8_512x512_80k_ade20k_20200615_014848-1f4929a3.pth'
    model = init_segmentor(config_file, checkpoint_file, device='cuda:2')
    model.eval()
    return model

def _single_test_torch(model):
    time_cost_list = []
    with torch.no_grad():
        for i in range(100):
            input = torch.randn(size=(1,3,512,512)).cuda()
            model.test_speed(input)
            torch.cuda.synchronize()
        print("begin test")
        for i in range(500):
            input = torch.randn(size=(4,3,512,512)).cuda()
            # input = [np.random.randint(255,size=(512,512,3)) for _ in range(4)]
            torch.cuda.synchronize()
            start = time.time()
            model.test_speed(input)
            # inference_segmentor(model,input)
            torch.cuda.synchronize()
            end = time.time()
            time_cost_list.append((end-start)*1000)
    print("mean time cost: %s milisecond(s)"%(np.mean(time_cost_list)))
    return np.mean(time_cost_list)
    
if __name__ == "__main__":
    model = get_mmseg_model()
    for i in range(3):
        _single_test_torch(model)