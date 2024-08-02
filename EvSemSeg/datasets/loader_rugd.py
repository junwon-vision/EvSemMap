import os
import random
import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

RUGD_ROOT = '/kjyoung/dataset/RUGD'
IMAGE_POSTFIX = 'RUGD_frames-with-annotations'
LABEL_POSTFIX = 'RUGD_annotations'
COLOR_POSTFIX = 'RUGD_annotation-colormap.txt'
REMAPPING_COLOR_POSTFIX = 'remapped_color.txt'

train_val_test_split = {
    "train": ['park-2', 'trail', 'trail-3', 'trail-4', 'trail-6', 'trail-9', 'trail-10', 'trail-11', 'trail-12', 'trail-14', 'trail-15', 'village'],
    "val": ['park-8', 'trail-5'],
    "test": ['creek', 'park-1', 'trail-7', 'trail-13'],
}
LOADER_PHASE = ['train', 'val', 'test']
# prep for preparation of this loader(development purpose)

from .loader_util import RemapOBJ

# RUGD+Rellis Remap V3

class RUGDLoader(data.Dataset):
    """
        RUGD Dataset Loader
    """
    
    def __init__(self, data_subset='train', shuffle=True, remap_version=3, partial_val=None):
        """__init__

        :param phase:
        :param shuffle:
        :param remapping:
        """
        assert data_subset in LOADER_PHASE

        self.phase = data_subset
        self.images = []
        self.annotations = []

        self.img_root_dir = os.path.join(RUGD_ROOT, IMAGE_POSTFIX)
        self.lbl_root_dir = os.path.join(RUGD_ROOT, LABEL_POSTFIX)
        
        
        target_names = train_val_test_split[self.phase]

        for target_name in target_names:
            target_image_dir = os.path.join(RUGD_ROOT, IMAGE_POSTFIX, target_name)
            target_label_dir = os.path.join(RUGD_ROOT, LABEL_POSTFIX, target_name)

            image_files = np.sort(os.listdir(target_image_dir))
            image_files = [os.path.join(target_name, f) for f in image_files if f.endswith(".png")]

            label_files = np.sort(os.listdir(target_label_dir))
            label_files = [os.path.join(target_name, f) for f in label_files if f.endswith(".png")]

            image_files, label_files = sorted(image_files), sorted(label_files)
            
            self.images.extend(image_files)
            self.annotations.extend(label_files)
        
        # Check Validity
        assert len(self.images) == len(self.annotations)
        for i, l in zip(self.images, self.annotations):
            assert os.path.basename(i) == os.path.basename(l)

        # Shuffle
        if shuffle:
            zipped_list = list(zip(self.images, self.annotations))
            random.shuffle(zipped_list)
            self.images, self.annotations = zip(*zipped_list)

        if partial_val is not None:
            assert 0 < partial_val and partial_val < len(self.images)
            self.images = self.images[:partial_val]
            self.annotations = self.annotations[:partial_val]

        self.data_size = len(self.images)
        print(f"RUGD Dataset Validity Checked : {self.phase} phase - {len(self.images)} image+label pairs found.")

        # Load ColorMap Info
        color_map = pd.read_csv(os.path.join(self.lbl_root_dir, COLOR_POSTFIX), sep=" ", header=None)
        color_map.columns = ["label_idx", "label", "R", "G", "B"]

        # self.label2id = {label : id for id, label in enumerate(color_map.label)}
        # self.id2label = {id : label for id, label in enumerate(color_map.label)}
        self.id2color = {id: [R, G, B] for id, (R, G, B) in enumerate(zip(color_map.R, color_map.G, color_map.B))}

        if remap_version != -1:
            self.remapping = RemapOBJ[remap_version]
            self.new_id2color = self.remapping['id2color']
            self.org_id2new_id = self.remapping['rugd_orgID2remapID']
            self.n_classes = self.remapping['n_class']
        else:
            self.remapping = None
            
# Documentation
# label2id {'void': 0, 'dirt': 1, 'sand': 2, 'grass': 3, 'tree': 4, 'pole': 5, 'water': 6, 'sky': 7, 'vehicle': 8, 'container/generic-object': 9, 'asphalt': 10, 'gravel': 11, 'building': 12, 'mulch': 13, 'rock-bed': 14, 'log': 15, 'bicycle': 16, 'person': 17, 'fence': 18, 'bush': 19, 'sign': 20, 'rock': 21, 'bridge': 22, 'concrete': 23, 'picnic-table': 24}
# id2label {0: 'void', 1: 'dirt', 2: 'sand', 3: 'grass', 4: 'tree', 5: 'pole', 6: 'water', 7: 'sky', 8: 'vehicle', 9: 'container/generic-object', 10: 'asphalt', 11: 'gravel', 12: 'building', 13: 'mulch', 14: 'rock-bed', 15: 'log', 16: 'bicycle', 17: 'person', 18: 'fence', 19: 'bush', 20: 'sign', 21: 'rock', 22: 'bridge', 23: 'concrete', 24: 'picnic-table'}
# id2color {0: [0, 0, 0], 1: [108, 64, 20], 2: [255, 229, 204], 3: [0, 102, 0], 4: [0, 255, 0], 5: [0, 153, 153], 6: [0, 128, 255], 7: [0, 0, 255], 8: [255, 255, 0], 9: [255, 0, 127], 10: [64, 64, 64], 11: [255, 128, 0], 12: [255, 0, 0], 13: [153, 76, 0], 14: [102, 102, 0], 15: [102, 0, 0], 16: [0, 255, 128], 17: [204, 153, 255], 18: [102, 0, 204], 19: [255, 153, 204], 20: [0, 102, 102], 21: [153, 204, 255], 22: [102, 255, 255], 23: [101, 101, 11], 24: [114, 85, 47]}
    def __len__(self):
        """__len__"""
        return self.data_size

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        image = Image.open(os.path.join(self.img_root_dir, self.images[index]))
        annotation = Image.open(os.path.join(self.lbl_root_dir, self.annotations[index]))

        annotation = np.array(annotation)

        annotation_2d = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.uint8)

        if self.remapping:
            for id, color in self.id2color.items():
                annotation_2d[(annotation == color).all(axis=-1)] = self.org_id2new_id[id]
        else:
            for id, color in self.id2color.items():
                annotation_2d[(annotation == color).all(axis=-1)] = id
        
        return self.transform(image, annotation_2d)
    
    def transform(self, img, lbl) :
        img, lbl = TF.to_tensor(img), torch.from_numpy(lbl)
        img = TF.normalize(img, mean=[103.939/255, 116.779/255, 123.68/255], std=[0.229, 0.224, 0.225])
        return img, lbl

    def label_transform(self, lbl, gt=True):
        """
        label_transform : to visualize
        :param lbl:
        """
        lbl_vis = torch.zeros((3, lbl.shape[0], lbl.shape[1])).float().cuda()
        
        if self.remapping:
            iterator = self.new_id2color.items()
            for id, color in iterator:
                # import pdb; pdb.set_trace()
                if (lbl == id).sum() > 0 :
                    lbl_vis[:,lbl == id] = torch.tensor(np.array(color)).float().unsqueeze(1).cuda() 	/ 255.
        
        else:
            iterator = self.id2color.items()
            for id, color in iterator:
                # import pdb; pdb.set_trace()
                if (lbl == id).sum() > 0 :
                    lbl_vis[:,lbl == id] = torch.tensor(np.array(color)).float().unsqueeze(1).cuda() 	/ 255.
        
        return lbl_vis