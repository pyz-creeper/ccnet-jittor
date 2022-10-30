import pickle
import numpy as np
from PIL import Image
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
        return len(self.img_info)
        
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
                    "img_dir":os.path.join(self.data_root,"images","training",name),
                    "ann_dir":os.path.join(self.data_root,"annotations","training",name)
                })
            if not self.is_train and name[0:9] != "ADE_train":
                # add a validation sample
                img_infos.append({
                    "label":label,
                    "img_dir":os.path.join(self.data_root,"images","validation",name),
                    "ann_dir":os.path.join(self.data_root,"annotations","validation",name)
                })
        return img_infos

    def __getitem__(self, index):
        img_info = self.img_infos[index]
        img = np.array(Image.open(img_info['img_dir']))
        ann = np.array(Image.open(img_info['ann_dir']))
        # TODO reshaped those img into 512 * 512 shape
        # add shapes to attribute?
        return img,ann

    def evaluate(self, results):
        '''
        list of numpy result img given by the model
        '''
        pass

