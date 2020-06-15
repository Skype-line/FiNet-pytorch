import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw
import os.path as osp
import numpy as np
import json
import random
from random import choice

from utils import seg1_to_seg8

class FiDataset(Dataset):
	"""Dataset for FiNet"""
	def __init__(self, opt):
		super(FiDataset, self).__init__()
		#base setting
		self.opt = opt
		# self.root = opt.dataroot
		self.exp_name = opt.exp_name
		self.mode = opt.mode
		self.stage = opt.stage
		# self.target_type = opt.target_type
		self.data_list = opt.data_list
		self.fine_height = opt.fine_height
		self.fine_width = opt.fine_width
		self.radius = opt.radius
		self.data_path = osp.join(opt.dataroot, opt.mode)
		self.transform = transforms.Compose([  \
				transforms.ToTensor(),   \
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #[-1,1]

		# load data list
		img_names = []
		with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
			for line in f.readlines():
				if (self.mode == 'exp'):#|(self.mode == 'test'):
					for i in range(8): # experiment: each input generates 5 different outputs 
						img_names.append(line.strip())
				else:
					img_names.append(line.strip())
		self.img_names = img_names
		print('Size of the dataset: %05d' % (len(img_names)))

	def name(self):
		return "FiDataset"

	def __getitem__(self, index):
		img_name = self.img_names[index]
		parse_name = img_name.replace('.jpg','.png')
		im_pil = Image.open(osp.join(self.data_path, 'image', img_name))
		im = transforms.ToTensor()(im_pil) # [0,1] 3*256*256 3*H*W Tensor
		im_pil.close()
		# if (self.mode == 'test') & (self.stage == 'stage2'):
		# 	im_parse = Image.open(osp.join(self.exp_name, 'result_images/stage1/test', parse_name))
		# else:
		im_parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
		parse_array = np.array(im_parse) # H*W
		im_parse.close()

		parse_head = (parse_array == 1).astype(np.float32)
		# 1 of 4 is target and others are contextual garments
		parse_hat = (parse_array == 4).astype(np.float32)
		parse_top_cloth = (parse_array == 5).astype(np.float32)
		parse_bottom_cloth = (parse_array == 6).astype(np.float32)
		parse_shoes = (parse_array == 7).astype(np.float32)

		phead = torch.from_numpy(parse_head) # [0,1] Tensor H*W
		phat = torch.from_numpy(parse_hat)
		ptop = torch.from_numpy(parse_top_cloth)
		pbottom = torch.from_numpy(parse_bottom_cloth)
		pshoes = torch.from_numpy(parse_shoes)
		p_items = {'hat':phat, 'top':ptop, 'bottom':pbottom, 'shoes':pshoes}

		im_hat = im * phat + (1 - phat)		# [0,1] 3*H*W Tensor
		im_top = im * ptop + (1 - ptop)
		im_bottom = im * pbottom + (1 - pbottom)
		im_shoes = im * pshoes + (1 - pshoes)
		im_items = {'hat':im_hat, 'top':im_top, 'bottom':im_bottom, 'shoes':im_shoes}

		bbox_name = img_name.replace('.jpg','.json')
		with open(osp.join(self.data_path, 'bbox_coordinate', bbox_name), 'r') as f:
			box = json.load(f) 
		
		flag = 1
		Target_Types = ['hat','top','bottom','shoes']
		while flag:
			# self.target_type = choice(Target_Types)
			self.target_type = 'top'
			target = p_items[self.target_type]
			coor = box[self.target_type]
			try:
				crop_target = target[coor['y_up']:coor['y_down'],coor['x_left']:coor['x_right']]
				flag = 0
				# print(crop_target.size())
				# print(crop_target)
				if crop_target.nelement() == 0:
					flag = 1
			except TypeError:
				flag = 1
		"""
		Encoder_C Input: context_garments, same for both 'stage1' and 'stage2'
			 import bounding box coordinates of 4 type: 'hat','top','bottom','shoes'
			 for crop & resize & pad to 3*256*256
		"""
		blank_im = torch.ones(3,self.fine_height,self.fine_width) # set target category image to all 1's
		for item_name in ['hat','top','bottom','shoes']:
			try:
				coor = box[item_name]
				crop_item = im_items[item_name][:,coor['y_up']:coor['y_down'],coor['x_left']:coor['x_right']] #[-1,1]
				crop_item = transforms.ToPILImage()(crop_item)
				ratio = self.fine_height / max(crop_item.size)
				new_size = tuple([int(x*ratio) for x in crop_item.size])
				resized_item = crop_item.resize(new_size, Image.ANTIALIAS)
				new_item = Image.new("RGB",(self.fine_width,self.fine_height),color=(255,255,255))
				new_item.paste(resized_item, ((self.fine_width-new_size[0])//2,
												(self.fine_height-new_size[1])//2))  # W*H [0,255]
				assert new_item.size == (self.fine_width,self.fine_height)
				new_item = self.transform(new_item) # [-1,1] 3*256*256  3*H*W Tensor
			except TypeError:
				new_item = blank_im
			im_items[item_name] = new_item

		# if one item not exist (e.g. hat) replace with mean value of hat 
		for key, item in im_items.items():
			if torch.equal(item, blank_im):
				mean = torch.from_numpy(np.load(osp.join(self.data_path,"%s_%s_mean_new.npy" % (key,self.mode))))
				im_items[key] = mean

		im_hat, im_top, im_bottom, im_shoes = im_items['hat'], im_items['top'], im_items['bottom'], im_items['shoes']

		if self.target_type == 'top':
			context_garments = torch.cat((im_hat,blank_im,im_bottom,im_shoes),0)  # 12*256*256 H*W [-1,1]
		elif self.target_type == 'bottom':
			context_garments = torch.cat((im_hat,im_top,blank_im,im_shoes),0)
		elif self.target_type == 'hat':
			context_garments = torch.cat((blank_im,im_top,im_bottom,im_shoes),0)
		elif self.target_type == 'shoes':
			context_garments = torch.cat((im_hat,im_top,im_bottom,blank_im),0)
		else:
			print('Inpainted type %s not exist, choose another one!' % self.target_type)


		"""
		Encoder_I Input: target,  'stage1': target shape & 'stage2': target image
		"""
		
		if self.stage == 'stage1':
			crop_target = transforms.ToPILImage()(crop_target.unsqueeze_(0))
			ratio = self.fine_height / max(crop_target.size)
			new_size = tuple([int(x*ratio) for x in crop_target.size])
			resized_target = crop_target.resize(new_size, Image.ANTIALIAS)
			new_target = Image.new("L",(self.fine_width,self.fine_height))
			new_target.paste(resized_target, ((self.fine_width-new_size[0])//2,
											(self.fine_height-new_size[1])//2))  # W*H [0,255]
			assert new_target.size == (self.fine_width,self.fine_height)
			target = self.transform(new_target)		 # 1*256*256  1*H*W ToTensor
		else:
			target = im_items[self.target_type]  # 3*128*128
		
		"""
		Generator Input:
			person representation: 'stage1': head shape + pose_map   &  'stage2':head image + human parse
			masked context: 	   'stage1': masked human parse      &  'stage2':masked human image
		GT: inpainted region:	   'stage1': resized human parse in the bbox &  'stage2':resized human image in the bbox
		"""
		im = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(im) #[-1,1]
		parse_human_c8 = seg1_to_seg8(parse_array) # 8*256*256 [0,1] tensor
		
		mask = torch.ones(1,self.fine_height, self.fine_width)
		coor = box[self.target_type] 
		mask[0,coor['y_up']:coor['y_down'],coor['x_left']:coor['x_right']] = 0
		mask_info = (coor['y_up'], coor['y_down'], coor['x_left'], coor['x_right'], coor['y_down']-coor['y_up'],coor['x_right']-coor['x_left']) # x/y coordinate & H,W of bbox

		if self.stage == 'stage1':
			phead = transforms.Normalize([0.5],[0.5])(phead.unsqueeze(0)) # [-1,1] Tensor 1*256*256

			# load pose points
			pose_name = img_name.replace('.jpg','_keypoints.json')
			with open(osp.join(self.data_path, 'pose', pose_name),'r') as f:
				pose_label = json.load(f)
				pose_data = pose_label['pose_keypoints']
				pose_data = np.array(pose_data)
				pose_data = pose_data.reshape((-1,3))
			point_num = pose_data.shape[0]
			pose_map = torch.zeros(point_num, self.fine_height, self.fine_width) # pose_map! [-1,1]
			r = self.radius
			im_pose = Image.new('L', (self.fine_width, self.fine_height))
			pose_draw = ImageDraw.Draw(im_pose)
			for i in range(point_num):
				one_map = Image.new('L', (self.fine_width, self.fine_height))
				draw = ImageDraw.Draw(one_map)
				pointx = pose_data[i,0]
				pointy = pose_data[i,1]
				if pointx > r and pointy > r:
					draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
					pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
				one_map = self.transform(one_map)
				pose_map[i] = one_map[0]
			im_pose = self.transform(im_pose) # just for visualization

			# if self.target_type == 'top':
			# 	parse_human_c8[2] == 0
			# 	parse_human_c8[5] == 0
			# elif self.target_type == 'bottom':
			# 	parse_human_c8[3] == 0
			# elif self.target_type == 'hat':
			# 	parse_human_c8[4] == 0
			# elif self.target_type == 'shoes':
			# 	parse_human_c8[7] == 0
			masked_parse = parse_human_c8 * mask  #[0,1] 8*256*256
			# inpainted_parse = parse_human_c8[:,coor['y_up']:coor['y_down'],coor['x_left']:coor['x_right']]
			# inpainted_parse = F.interpolate(inpainted_parse.unsqueeze(0),size=(self.fine_height, self.fine_width),mode='bilinear',align_corners=True)[0] # upsample to 8*256*256
			# inpainted_parse = torch.argmax(inpainted_parse, dim=0)
			parse_human_c1 = torch.argmax(parse_human_c8, dim=0)

			pp_represent = torch.cat((phead,pose_map),0) 	 	 # 19*256*256
			masked_context = masked_parse						 # 8*256*256
			ground_truth = parse_human_c1					 # 256*256
		else:
			im_head = im * phead + (1 - phead)
			# if self.target_type == 'top':
			# 	masked_im = im * (1 - parse_human_c8[2] - parse_human_c8[5]) 				#[-1,1] 3*256*256
			# elif self.target_type == 'bottom':
			# 	masked_im = im * (1 - parse_human_c8[3]) 				#[-1,1] 3*256*256
			# elif self.target_type == 'hat':
			# 	masked_im = im * (1 - parse_human_c8[4]) 				#[-1,1] 3*256*256
			# elif self.target_type == 'shoes':
			# 	masked_im = im * (1 - parse_human_c8[7]) 				#[-1,1] 3*256*256
			masked_im = im * mask				#[-1,1] 3*256*256
			# inpainted_im = im[:,coor['y_up']:coor['y_down'],coor['x_left']:coor['x_right']]
			# inpainted_im = F.interpolate(inpainted_im.unsqueeze(0),size=(self.fine_height_G, self.fine_width_G),mode='bilinear',align_corners=True)[0] # upsample to 3*256*256

			pp_represent = torch.cat((im_head,parse_human_c8),0) # 11*256*256
			masked_context = masked_im							 # 3*256*256
			ground_truth = im						 # 3*256*256
		
		result = {
			'img_name':img_name,
			'target':target,
			'context_garments':context_garments,
			'pp_represent':pp_represent,
			'mask':mask,
			'mask_info':mask_info,
			'masked_context':masked_context,
			'ground_truth':ground_truth,
			'target_type':self.target_type
			# 'pose':im_pose
			}
		return result

	def __len__(self):
		return len(self.img_names)


class FiDataLoader(object):
	def __init__(self, opt, dataset):
		super(FiDataLoader, self).__init__()

		if opt.shuffle:
			data_sampler = RandomSampler(dataset)
		else:
			data_sampler = None

		if opt.mode == 'train':
			self.data_loader = DataLoader(
					dataset, batch_size=opt.batch_size, shuffle=(data_sampler is None),
					num_workers=opt.workers, pin_memory=True, sampler=data_sampler)
		else:
			self.data_loader = DataLoader(
					dataset, batch_size=opt.batch_size, shuffle=False,
					num_workers=opt.workers, pin_memory=True, sampler=data_sampler)
		self.dataset = dataset
		self.data_iter = self.data_loader.__iter__()

	def next_batch(self):
		try:
			batch = self.data_iter.__next__()
		except StopIteration:
			self.data_iter = self.data_loader.__iter__()
			batch = self.data_iter.__next__()

		return batch

if __name__ == "__main__":
	
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_name", default = "Exp1")
	parser.add_argument("--dataroot", default = "dataset")
	parser.add_argument("--mode", default = "train")
	parser.add_argument("--stage", default = "stage2")
	parser.add_argument('--target_type', default='bottom')
	parser.add_argument("--data_list", default = "train.txt")
	parser.add_argument("--fine_width", type=int, default = 256)
	parser.add_argument("--fine_height", type=int, default = 256)
	parser.add_argument("--radius", type=int, default = 3)
	parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
	parser.add_argument('-b', '--batch_size', type=int, default=16)
	parser.add_argument('-j', '--workers', type=int, default=1)

	opt = parser.parse_args()
	dataset = FiDataset(opt)
	data_loader = FiDataLoader(opt, dataset)
	print("Check the dataset for stage %s!" % opt.stage)
	print('Size of the dataset: %05d, dataloader: %04d' \
			% (len(dataset), len(data_loader.data_loader)))
	first_item = dataset.__getitem__(0)
	first_batch = data_loader.next_batch()

	from IPython import embed; embed()
