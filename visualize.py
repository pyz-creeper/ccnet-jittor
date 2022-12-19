import cv2
import os
from PIL import Image
import numpy as np
import jittor
jittor.flags.use_cuda = 1
from networks.ccnet import CCnet, VAN_CCnet
from datasets.data_pipeline import Pipeline
from jittor.misc import make_grid

CLASSES = (
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

PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]


def _show_result_np(result):
    '''
    input result should be h*w*1
    range from 0 ~ 149
    '''
    result_colored = np.zeros((result.shape[0],result.shape[1],3))
    for idx, color in enumerate(PALETTE):
        result_colored[result == idx] = color
    return result_colored

def inferece_result(model, img, datapipeline):
    img, _ = datapipeline(img,None)
    img = jittor.array(img.transpose(2,0,1))
    img = jittor.unsqueeze(img,0)
    with jittor.no_grad():
        output,_ = model(img)
        seg_pred = np.array(output.argmax(dim=1)[0].numpy(), dtype=np.int32).squeeze(0)
    return _show_result_np(seg_pred)

def show_result(imgs,gt_labels,pretrained_ckpt,backbone_type,save_dir):
    '''
    imgs: list[str] list of img's path
    pretrained_ckpt: pkl file path for model to load
    backbone_type: VAN or Resnet
    '''
    if backbone_type == "van":
        model = VAN_CCnet(attention_block="vanilla", recurrence=2, pretrained=False)
    else:
        model = CCnet(attention_block="vanilla", recurrence=2, pretrained=False)
    model.load(pretrained_ckpt)
    datapipeline = Pipeline(train=False)
    for img_path,label_path in zip(imgs,gt_labels):
        img = cv2.imread(img_path)
        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        colored_result = inferece_result(model, img, datapipeline)
        img = np.array(img)[:,:,::-1]
        label = np.array(label)
        label[label==0] = 1
        label -= 1
        colored_gt = _show_result_np(label)
        im_name = os.path.basename(img_path)
        Image.fromarray((img).astype(np.uint8)).save(os.path.join(save_dir,"ori_"+im_name[:-3]+"png"))
        Image.fromarray((colored_result).astype(np.uint8)).save(os.path.join(save_dir,"pre_"+im_name[:-3]+"png"))
        Image.fromarray((colored_result/2+img/2).astype(np.uint8)).save(os.path.join(save_dir,"mix_"+im_name[:-3]+"png"))
        Image.fromarray((colored_gt/2+img/2).astype(np.uint8)).save(os.path.join(save_dir,"gt_"+im_name[:-3]+"png"))
        # cv2.imwrite(os.path.join(save_dir,im_name),colored_result.astype(np.uint8))
        

if __name__ == "__main__":
    show_result(["./ADEChallengeData2016/images/validation/ADE_val_00000001.jpg"],
                ["./ADEChallengeData2016/annotations/validation/ADE_val_00000001.png"],
                "./saves/ckpts/1215_1e-2_16_train_ccnet_resnet_epoch80000.pkl",
                "resnet","./tmp")