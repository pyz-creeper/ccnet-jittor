import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import jittor
jittor.flags.use_cuda = 1
from datasets.dataloaders import ADE20k
from networks.resnet import Resnet101
from networks.ccnet import CCnet
from evaluate.test import test_single_gpu,get_confusion_matrix
from tqdm import tqdm
import numpy as np



if __name__ == "__main__":
    model = CCnet(attention_block="vanilla", recurrence=2)
    model.load("./saves/ckpts/1208_final_16_train_ccnet_resnet_epoch16000.pkl")
    test_single_gpu(model)
    # dataset = ADE20k(1,"./ADEChallengeData2016",train=False)
    # confusion_matrix = np.zeros((151,151))
    # with jittor.no_grad():
    #     for index, (img, ann) in tqdm(enumerate(dataset)):
    #         ann = ann.numpy()
    #         confusion_matrix += get_confusion_matrix(ann, ann, 151)
    #         break
    #     pos = confusion_matrix.sum(1)
    #     res = confusion_matrix.sum(0)
    #     tp = np.diag(confusion_matrix)
    #     IU_array = (tp / np.maximum(1.0, pos + res - tp))
    #     mean_IU = IU_array.mean()
    #     print("mean_IoU:",mean_IU)

