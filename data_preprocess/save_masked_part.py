import cv2
import torch
import os
import os.path as osp
import numpy as np
import sys
from PIL import Image
import json
import torchvision.transforms as transforms

sys.path.append('..')
from utils import seg1_to_seg8, seg8_to_seg1

def main():
	fine_height_G = 256
	fine_width_G = 256
	target_type = 'top'

	mode = 'test'
	data_list = mode+'.txt'
	img_save_dir = osp.join(mode, 'masked_im_'+target_type)
	parse_save_dir = osp.join(mode, 'masked_parse_'+target_type)

	if not osp.exists(img_save_dir):
		os.makedirs(img_save_dir)
	if not osp.exists(parse_save_dir):
		os.makedirs(parse_save_dir)

	img_names = []
	with open(osp.join('.', data_list), 'r') as f:
		for line in f.readlines():
			img_names.append(line.strip())

	for img_name in img_names:
		im = Image.open(osp.join(mode, 'image', img_name))
		im = transforms.ToTensor()(im) # [0,1] 3*256*256 3*H*W Tensor

		parse_name = img_name.replace('.jpg','.png')
		im_parse = Image.open(osp.join(mode, 'image-parse', parse_name))
		parse_array = np.array(im_parse) # H*W
		parse_human_c8 = seg1_to_seg8(parse_array) # 8*256*256 [0,1] tensor

		bbox_name = img_name.replace('.jpg','.json')
		with open(osp.join(mode, 'bbox_coordinate', bbox_name), 'r') as f:
			box = json.load(f)
		mask = torch.zeros(fine_height_G, fine_width_G)
		coor = box[target_type] 
		mask[coor['y_up']:coor['y_down'],coor['x_left']:coor['x_right']] = 1

		masked_im = im * (1 - parse_human_c8[2] - parse_human_c8[5])
		masked_im_show = transforms.ToPILImage()(masked_im)
		masked_im_show.save(osp.join(img_save_dir, img_name))

		masked_parse_c8 = parse_human_c8 * (1 - mask)  #[0,1] 8*256*256
		masked_parse_c1 = seg8_to_seg1(masked_parse_c8)
		masked_parse_save = np.array(masked_parse_c1)
		cv2.imwrite('{}/{}.png'.format(parse_save_dir, img_name[:-4]), masked_parse_save)

if __name__ == '__main__':
	main()