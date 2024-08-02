import os
import torch, cv2
import numpy as np
from torch.utils import data
from PIL import Image
import random
import torchvision.transforms.functional as TF
import pdb

from .loader_util import (
    rellis_orgid2color, rellis_orgid_compact20_to_color, rellis_orgid2_orgid_compact20,
	RemapOBJ,
)

# Corrupted Labels
# 00001: frame001774-1581623967_749
# 00002: frame002177-1581797368_109

rellis_split_dict = {
	"-0": ['00001', '00002', '00003', '00004'],
	"-1": ['00000', '00002', '00003', '00004'],
	"-2": ['00000', '00001', '00003', '00004'],
	"-3": ['00000', '00001', '00002', '00004'],
	"-4": ['00000', '00001', '00002', '00003'],
	"0": ['00000'],
	"1": ['00001'],
	"2": ['00002'],
	"3": ['00003'],
	"4": ['00004'],
}

LOADER_PHASE = ['train', 'val', 'test']
RELLIS_ROOT = '/data/Rellis-3D'
RGB_POSTFIX = 'pylon_camera_node'
LBL_POSTFIX = 'pylon_camera_node_label_color'

class Rellis3DLoader(data.Dataset):
	"""
		Rellis-3D Dataset Loader
	"""
	

	
	def __init__(self,
			data_subset,
			ds_factor = None,
			shuffle=True, partial_val=None, only_paired=True, remap_version=3
		):
		"""__init__

		:param dataset:
		:param img_size:
		"""
		# self.n_classes = 10
		self.dataset = rellis_split_dict[data_subset]
		self.images = []
		self.annotations = []
		
		assert only_paired == True
		# Load Img, Lbl file names
		for data in self.dataset:
			img_dir = os.path.join(RELLIS_ROOT, data, RGB_POSTFIX)
			lbl_dir = os.path.join(RELLIS_ROOT, data, LBL_POSTFIX)

			img_files = np.sort(os.listdir(img_dir))
			img_keys = sorted([f[:-4] for f in img_files if f.endswith(".jpg")])

			lbl_files = np.sort(os.listdir(lbl_dir))
			lbl_keys = sorted([f[:-4] for f in lbl_files if f.endswith(".png")])
			
			for lbl_key in lbl_keys:
				if lbl_key in img_keys:
					pass
				else:
					raise Exception(f"Some Label has no counterparts; {lbl_key}")
			
			lbl_keys = [f for f in lbl_keys if (f != 'frame001774-1581623967_749' and f != 'frame002177-1581797368_109')] # Label File Corrupted in 77Server

			img_files = [os.path.join(img_dir, f"{f}.jpg") for f in lbl_keys]
			lbl_files = [os.path.join(lbl_dir, f"{f}.png") for f in lbl_keys]

			self.images.extend(img_files)
			self.annotations.extend(lbl_files)
			print(f"DATA SPLIT {data} @ {len(img_files)} images / {len(lbl_files)} labels found.")
		
		# Check Validity
		assert len(self.images) == len(self.annotations)
		for i, l in zip(self.images, self.annotations):
			assert os.path.basename(i)[:-4] == os.path.basename(l)[:-4]

		# Shuffle
		if shuffle:
			zipped_list = list(zip(self.images, self.annotations))
			random.shuffle(zipped_list)
			self.images, self.annotations = zip(*zipped_list)
		
		# Partial Validation for Experiments
		if partial_val is not None:
			assert 0 < partial_val and partial_val < len(self.images)
			self.images = self.images[:partial_val]
			self.annotations = self.annotations[:partial_val]
		
		self.data_size = len(self.images)
		print(f"RELLIS-3D Dataset Validity Checked : {len(self.images)} image+label pairs found.")

		self.original_id2color = rellis_orgid2color
		self.ds_factor = ds_factor
		self.remapOBJ = RemapOBJ[remap_version] if remap_version != -1 else None

	# DATA SPLIT 00000 @ 2847 images / 1200 labels found.
	# DATA SPLIT 00001 @ 2319 images / 1010 labels found.
	# DATA SPLIT 00002 @ 4147 images / 1443 labels found.
	# DATA SPLIT 00003 @ 2184 images / 581 labels found.
	# DATA SPLIT 00004 @ 2059 images / 875 labels found.
	# => if only_paird == True, Filtering Images

	def __len__(self):
		"""__len__"""
		return self.data_size

	def __getitem__(self, index):
		"""__getitem__

		:param index:
		"""
		image = Image.open(self.images[index]) # (1920, 1200)
		annotation = Image.open(self.annotations[index]) # (1920, 1200)
		
		if self.ds_factor is not None:
			width, height = image.size
			resized_size = (int(width / self.ds_factor), int(height / self.ds_factor))
			image = image.resize(resized_size)

		annotation = np.array(annotation) # (1200, 1920, 3)

		if annotation.shape == ():
			print(f"{annotation.shape} ?? : {self.images[index]} / {self.annotations[index]}")
			
		annotation_2d = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.uint8)

		for id, color in self.original_id2color.items():
			if self.remapOBJ:
				annotation_2d[(annotation == color).all(axis=-1)] = self.remapOBJ['rellis_orgID2remapID'][id]
			else:
				annotation_2d[(annotation == color).all(axis=-1)] = rellis_orgid2_orgid_compact20[id]
		
		if self.ds_factor is not None:
			annotation_2d = cv2.resize(annotation_2d, dsize=resized_size, interpolation=cv2.INTER_NEAREST)
		
		return self.transform(image, annotation_2d)

	def label_transform(self, lbl):
		"""
		label_transform : to visualize
		:param lbl:
		"""
		
		# 0 : void, sky
		# 1 : vehicle, person
		# 2 : pole, barrier, object, log, fence, building
		# 3 : asphalt, concrete
		# 4 : grass
		# 5 : dirt, mud
		# 6 : puddle
		# 7 : rubble, water
		# 8 : tree
		# 9 : bush

		lbl_vis = torch.zeros( (3, lbl.shape[0], lbl.shape[1])).float().cuda()

		if self.remapOBJ:
			for id_, color_ in self.remapOBJ['id2color'].items():
				if (lbl == id_).sum() > 0 :
					lbl_vis[:, lbl == id_] = torch.tensor(np.array(color_)).float().unsqueeze(1).cuda() 	/ 255.
		else:
			for id, color in rellis_orgid_compact20_to_color.items():
				if (lbl == id).sum() > 0 :
					lbl_vis[:, lbl == id] = torch.tensor(np.array(color)).float().unsqueeze(1).cuda() 	/ 255.

		return lbl_vis
	
	def transform(self, img, lbl) :
		# resize = transforms.Resize(size=self.size)
		# img = resize(img)
		img, lbl = TF.to_tensor(img), torch.from_numpy(lbl)
		img = TF.normalize(img, mean=[103.939/255, 116.779/255, 123.68/255], std=[0.229, 0.224, 0.225])

		return img, lbl