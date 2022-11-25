import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
from tensorboardX import SummaryWriter

# TODO: change this to a argparser format
learning_rate = 2e-3
batch_size = 1
global_step = 160000 // batch_size
eval_gap = 8000 // batch_size
save_every = 8000 // batch_size

def train():
    model = CCnet()
    model.train()
    dataset = ADE20k(batch_size,"./ADEChallengeData2016",train=True,shuffle=True)
    criterion = CriterionDSN()
    optimizer = nn.SGD(model.parameters(),learning_rate,0.9,0.0005)
    writer = SummaryWriter("./saves/1125_train")
    step = 0
    num_epoches = global_step // len(dataset) + 1
    for epoch in range(num_epoches):
        for batch_idx, (img, ann) in tqdm(enumerate(dataset)):
            out,out_dsn = model(img)
            loss = criterion([out,out_dsn],ann)
            optimizer.lr = learning_rate * ((1-step/global_step)**0.9)
            writer.add_scalar("loss",loss.numpy(),global_step=step)
            optimizer.step(loss)
            step += 1
            jt.sync_all()
            jt.gc()
            if step%eval_gap == 0:
                test_single_gpu(model)
                model.train()
            if step%save_every == 0:
                model.save("./saves/1125_train/train_ccnet_resnet_epoch%d.pkl"%(step))
            if step > global_step :
                return


if __name__ == "__main__":
    train()