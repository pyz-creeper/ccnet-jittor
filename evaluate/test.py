import jittor
from datasets.dataloaders import ADE20k
from networks.resnet import Resnet101
from networks.ccnet import CCHead
import numpy as np

def get_confusion_matrix(gt_label, pred_label, class_num):
    gt_label = gt_label.reshape(-1)
    pred_label = pred_label.reshape(-1)
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix

def test_single_gpu(model,class_num=150):
    model.eval()
    dataset = ADE20k(1,"./ADEChallengeData2016",train=False) # load val set!
    confusion_matrix = np.zeros((class_num,class_num))
    with jittor.no_grad():
        for index, (img, ann) in enumerate(dataset):
            ann = ann.numpy().astype(np.uint8)
            output = model(img)
            seg_pred = np.asarray(np.argmax(output.numpy(), axis=1), dtype=np.uint8)
            confusion_matrix += get_confusion_matrix(ann, seg_pred, class_num)
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()
        print("mean_IoU:",mean_IU)


if __name__ == "__main__":
    model = Resnet101()
    head = CCHead()
    x = jittor.random([10,3,512,683])
    x = model(x)
    out = head(x)
    print(out.shape)
