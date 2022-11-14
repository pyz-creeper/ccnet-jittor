import random
import numpy as np
from PIL import Image
import cv2
from jittor.dataset import Dataset
import os
from tqdm import tqdm

# ADE20K 2016 dataset
class ADE20k(Dataset):
    # TODO batch_size default value
    # TODO data_root default value
    # TODO ask lzy how to write a dataloader(just kidding)
    def __init__(self, batch_size, data_root, transform_pipeline = None,train=True , shuffle=False):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_index = []
        self.is_train = train
        img_file_path = os.path.join(data_root,"sceneCategories.txt")
        self.img_infos = self.parse_category_file(img_file_path)
        self.pipeline = transform_pipeline
        self.short_side_length = [300,375,450,525,600]


    def __len__(self):
        return len(self.img_infos)
        
    def parse_category_file(self, img_file_path):
        """
        parse the image info from sceneCategories.txt given by the dataset
        if we are training, the returned info will be in training set infos. 
        if not, then validation infos will be returned
        """
        with open(img_file_path,'r') as f:
            lines = f.readlines()
        img_infos = []
        for line in lines:
            name, label = line.split(' ')
            if self.is_train and name[0:9] == "ADE_train":
                # add a train sample
                img_infos.append({
                    "label":label,
                    "img_dir":os.path.join(self.data_root,"images","training",name+".jpg"),
                    "ann_dir":os.path.join(self.data_root,"annotations","training",name+".png")
                })
            if not self.is_train and name[0:9] != "ADE_train":
                # add a validation sample
                img_infos.append({
                    "label":label,
                    "img_dir":os.path.join(self.data_root,"images","validation",name+".jpg"),
                    "ann_dir":os.path.join(self.data_root,"annotations","validation",name+".png")
                })
        self.shuffle_index = np.arange(len(img_infos))
        if self.shuffle:
            np.random.shuffle(self.shuffle_index)
        return img_infos

    def __getitem__(self, index):
        img_info = self.img_infos[self.shuffle_index[index]]
        ini_img = cv2.imread(img_info['img_dir'])
        ini_ann = cv2.imread(img_info['ann_dir'],cv2.IMREAD_GRAYSCALE)
        if self.is_train:
            ini_h, ini_w = ini_ann.shape
            # control cuda memory
            for i in range(4,0,-1):
                a = random.randint(0,i)
                short_len = min(self.short_side_length[a],ini_h,ini_w)
                if ini_w > ini_h: # ini_h => short_len
                    new_shape = (int(ini_w*short_len/ini_h),short_len)
                else:
                    new_shape = (short_len,int(ini_h*short_len/ini_w))
                if new_shape[0] * new_shape[1] < 350000:
                    break
            img = cv2.resize(ini_img,new_shape)
            img = np.array(img.transpose(2,0,1)[::-1]).astype(np.float32)
            img /= 255.0
            ann = cv2.resize(ini_ann,new_shape,interpolation=cv2.INTER_NEAREST)
            ann = np.array(ann).astype(np.uint8)
            return img,ann
        else:
            img = np.array(ini_img.transpose(2,0,1)[::-1]).astype(np.float32)
            img /= 255.0
            return img,ini_ann



if __name__ == "__main__":
    dataset = ADE20k(1,"../ADEChallengeData2016",train=True)
    max_h,max_w = 0,0
    for index,(img,ann) in tqdm(enumerate(dataset)):
        _,imh,imw = ann.shape
        if max_h*max_w < imh*imw:
            max_h,max_w = imh,imw
    print(max_h,max_w)
