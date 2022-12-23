import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jittor as jt
jt.flags.use_cuda=1
import numpy as np
from datasets.data_pipeline import Pipeline
from networks.ccnet import get_model
import time

def _single_test_jittor(model):
    test_random_samples = [jt.randn(1,3,512,512) for _ in range(100)]
    time_cost_list = []
    with jt.no_grad():
        # run 100 times forward
        for i in range(100):
            model(test_random_samples[i])
            jt.sync_all(True)
            jt.gc()
        # run 500 for test
        for i in range(500):
            input = jt.randn(4,3,512,512)
            start = time.time()
            pred = model(input)
            pred[0].sync()
            pred[1].sync()
            jt.sync_all(True)
            end = time.time()
            jt.gc()
            time_cost_list.append((end-start)*1000)
    print("mean time cost: %s milisecond(s)"%(np.mean(time_cost_list)))
    return np.mean(time_cost_list)


def speed_test_jt():
    model = get_model("resnet","./saves/ckpts/1215_1e-2_16_train_ccnet_resnet_epoch80000.pkl")
    model.eval()
    for i in range(5):
        res = _single_test_jittor(model)
        
if __name__ == "__main__":
    speed_test_jt()