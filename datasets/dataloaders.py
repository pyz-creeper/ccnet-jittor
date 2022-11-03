import pickle
import numpy as np
from PIL import Image
import cv2
from jittor.dataset import Dataset
import os

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
        self.is_train = train
        img_file_path = os.path.join(data_root,"sceneCategories.txt")
        self.img_infos = self.parse_category_file(img_file_path)
        self.pipeline = transform_pipeline


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
        return img_infos

    def __getitem__(self, index):
        img_info = self.img_infos[index]
        # img = cv2.resize(cv2.imread(img_info['img_dir']),[683,512])
        # img = np.array(img.transpose(2,0,1)[::-1]).astype(np.float32)

        # ann = cv2.resize(cv2.imread(img_info['ann_dir'],cv2.IMREAD_GRAYSCALE),[683,512])
        # ann = np.array(ann).astype(np.uint8)
        img = np.array(Image.open(img_info['img_dir'])).astype(np.float32).transpose(2,0,1)
        img /= 255.0
        ann = np.array(Image.open(img_info['ann_dir']))
        
        # TODO reshaped those img into 512 * 512 shape
        # add shapes to attribute?
        return img,ann

    def evaluate(self, results):
        '''
        list of numpy result img given by the model
        '''
        pass

