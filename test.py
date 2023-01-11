import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jittor
jittor.flags.use_cuda = 1
from datasets.dataloaders import ADE20k
from networks.resnet import Resnet101
from networks.ccnet import CCnet, VAN_CCnet
from evaluate.test import test_single_gpu,get_confusion_matrix
from tqdm import tqdm
import numpy as np
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, help="checkpoint directory")
    parser.add_argument("--recurrence", type=int, default=2, help="cc attention recurrence")
    parser.add_argument("--flip", type=bool, default=True, help="if flip the input and evaluate together")
    parser.add_argument("--attention_block", type=str, default='vanilla',
                        choices=['vanilla', 'neighborhood', 'dilated'], help="attention block type")
    parser.add_argument("--model_backbone", type=str, default='resnet',
                        choices=['resnet', 'van'], help="backbone")
    opt = parser.parse_args()
    return opt

def test(opt):
    if opt.model_backbone == 'resnet':
        model = CCnet(opt.attention_block, opt.recurrence,pretrained=False)
    else:
        model = VAN_CCnet(opt.attention_block, opt.recurrence, pretrained=False)
    model.load(opt.ckpt_dir)
    model.eval()
    test_single_gpu(model, scales = [1.0], flip=opt.flip)


if __name__ == "__main__":
    # model = CCnet(attention_block="vanilla", recurrence=2, pretrained=False)
    # model.load("./saves/ckpts/1215_1e-2_16_train_ccnet_resnet_epoch80000.pkl")
    # test_single_gpu(model,scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],flip=True)
    opt = parse_opt()
    test(opt)
