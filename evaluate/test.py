import jittor as jt
from datasets.dataloaders import ADE20k
from datasets.data_pipeline import Pipeline
import numpy as np
from tqdm import tqdm

CLASSES = ('background',
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
    'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
    'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
    'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
    'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
    'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
    'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
    'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
    'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
    'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
    'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
    'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
    'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
    'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
    'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
    'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
    'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
    'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
    'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
    'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
    'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
    'clock', 'flag')

def get_confusion_matrix(gt_label, pred_label, class_num):
    index = (gt_label * class_num + pred_label).astype('int32').reshape(-1)
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
    dataset = ADE20k(1,"./ADEChallengeData2016",train=False, transform_pipeline=Pipeline(train=False)) # load val set!
    confusion_matrix = np.zeros((class_num,class_num))
    with jt.no_grad():
        for index, (img, ann) in tqdm(enumerate(dataset)):
            ann = ann.numpy().astype(np.int32)
            output,_ = model(img)
            seg_pred = np.asarray(np.argmax(output.numpy(), axis=1), dtype=np.int32)
            confusion_matrix += get_confusion_matrix(ann, seg_pred, class_num)
            jt.sync_all()
            jt.gc()
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()
        print("mean_IoU:",mean_IU)

def get_val_set_loss(model,class_num=151):
    pass


# if __name__ == "__main__":
#     dataset = ADE20k(1,"../ADEChallengeData2016",train=False)
#     confusion_matrix = np.zeros((150,150))
#     with jittor.no_grad():
#         for index, (img, ann) in tqdm(enumerate(dataset)):
#             confusion_matrix += get_confusion_matrix(ann, ann, 150)
#             break
#         pos = confusion_matrix.sum(1)
#         res = confusion_matrix.sum(0)
#         tp = np.diag(confusion_matrix)

#         IU_array = (tp / np.maximum(1.0, pos + res - tp))
#         mean_IU = IU_array.mean()
#         print("mean_IoU:",mean_IU)
