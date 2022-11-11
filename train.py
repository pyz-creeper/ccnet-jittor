import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

from networks.ccnet import CCnet
from datasets.dataloaders import ADE20k
from loss.loss import CriterionDSN
import jittor as jt
if jt.has_cuda:
    jt.flags.use_cuda = 1 
from jittor import nn
from tqdm import tqdm


def train():
    model = CCnet()
    model.train()
    dataset = ADE20k(1,"./ADEChallengeData2016",train=True)
    criterion = CriterionDSN()
    optimizer = nn.SGD(model.parameters(),1e-2,0.9,0.0005)
    for batch_idx, (img, ann) in tqdm(enumerate(dataset)):
        out, out_dsn = model(img)
        loss = criterion([out,out_dsn],ann)
        # TODO implement power lr decay
        optimizer.step(loss.sum())
        print(batch_idx)


if __name__ == "__main__":
    train()