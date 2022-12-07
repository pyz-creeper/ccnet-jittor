import os
import argparse
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
# learning_rate = 2e-3
# batch_size = 16
# global_step = 80000
# eval_gap = 8000
# save_every = 8000

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default="2e-3", help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--iters", type=int, default=80000, help="total train iterations")
    parser.add_argument("--save_every", type=int, default=8000, help="save gap")
    parser.add_argument("--attention_block", type=str, default='vanilla',
                        choices=['vanilla', 'neighborhood', 'dilated'], help="attention block type")

    opt = parser.parse_args()
    return opt


def train(opt):
    model = CCnet()
    model.train()
    dataset = ADE20k(opt.batch_size, "./ADEChallengeData2016", train=True, shuffle=True, )
    criterion = CriterionDSN()
    optimizer = nn.SGD(model.parameters(), opt.lr, 0.9, 0.0005)
    writer = SummaryWriter("./saves/1205_aug_16")
    step = 0
    num_epochs = opt.iters * opt.batch_size // len(dataset) + 1
    for epoch in range(num_epochs):
        for batch_idx, (img, ann) in tqdm(enumerate(dataset)):
            out_m, out_aux = model(img)
            loss = criterion([out_m,out_aux],ann)
            optimizer.lr = opt.lr * (max((1 - step / opt.iters), 1e-4) ** 0.9)
            writer.add_scalar("loss", loss.numpy(), global_step=step)
            optimizer.step(loss)
            step += 1
            jt.sync_all()
            jt.gc()
            if step % opt.save_every == 0:
                model.save("./saves/1205_aug_16/train_ccnet_resnet_epoch%d.pkl"%(step))
            if step > opt.iters :
                return


if __name__ == "__main__":
    opt = parse_opt()
    train(opt)