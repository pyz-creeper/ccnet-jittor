import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

from networks.ccnet import CCnet
from datasets.dataloaders import ADE20k
from loss.loss import CriterionDSN
from evaluate.test import test_single_gpu
import jittor as jt
if jt.has_cuda:
    jt.flags.use_cuda = 1 
from jittor import nn
from tqdm import tqdm
import time

learning_rate = 1e-2
global_step = 10000
eval_gap = 800
save_every = 1000

def train():
    model = CCnet()
    model.train()
    dataset = ADE20k(1,"./ADEChallengeData2016",train=False,shuffle=True)
    criterion = CriterionDSN()
    optimizer = nn.SGD(model.parameters(),learning_rate,0.9,0.0005)
    step = 0
    num_epoches = global_step//len(dataset) + 1
    for epoch in range(num_epoches):
        for batch_idx, (img, ann) in tqdm(enumerate(dataset)):
            out = model(img)
            loss = criterion([out],ann)
            optimizer.lr = learning_rate * ((1-step/global_step)**0.9)
            print(loss)
            optimizer.step(loss)
            optimizer.zero_grad()
            step += 1
            # if step%eval_gap == 0:
            #     test_single_gpu(model)
            #     model.train()
            if step%save_every == 0:
                model.save("./saves/ccnet_resnet_epoch%d.pkl"%(step))


if __name__ == "__main__":
    train()