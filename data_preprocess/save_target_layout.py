import cv2
import torch
import os
import os.path as osp
import numpy as np
import sys
from PIL import Image
import json
import torchvision.transforms as transforms

def main():
	fine_height = 16
	fine_width = 16
	target_type = 'top'

	MODE = ['exp']
	for mode in MODE:
		data_list = mode+'.txt'
		layout_save_dir = osp.join(mode, 'layout1_'+target_type)

		if not osp.exists(layout_save_dir):
			os.makedirs(layout_save_dir)

		img_names = []
		with open(osp.join('.', data_list), 'r') as f:
			for line in f.readlines():
				img_names.append(line.strip())

		for img_name in img_names:
			parse_name = img_name.replace('.jpg','.png')
			im_parse = Image.open(osp.join(mode, 'image-parse', parse_name))
			parse_array = np.array(im_parse) # H*W
			# print(parse_array)
			im_parse.close()

			# parse_hat = (parse_array == 4).astype(np.float32)
			parse_top_cloth = (parse_array == 5).astype(np.float32)
			# parse_bottom_cloth = (parse_array == 6).astype(np.float32)
			# parse_shoes = (parse_array == 7).astype(np.float32)

			# phat = torch.from_numpy(parse_hat)
			ptop = torch.from_numpy(parse_top_cloth)
			# pbottom = torch.from_numpy(parse_bottom_cloth)
			# pshoes = torch.from_numpy(parse_shoes)
			# p_items = {'hat':phat, 'top':ptop, 'bottom':pbottom, 'shoes':pshoes}

			bbox_name = img_name.replace('.jpg','.json')
			with open(osp.join(mode, 'bbox_coordinate', bbox_name), 'r') as f:
				box = json.load(f)
				coor = box[target_type]
			crop_target = ptop[coor['y_up']:coor['y_down'],coor['x_left']:coor['x_right']]
			crop_target = transforms.ToPILImage()(crop_target.unsqueeze_(0))
			ratio = fine_height / max(crop_target.size)
			new_size = tuple([int(x*ratio) for x in crop_target.size])
			resized_target = crop_target.resize(new_size, Image.ANTIALIAS)
			new_target = Image.new("L",(fine_width,fine_height))
			new_target.paste(resized_target, ((fine_width-new_size[0])//2,
											(fine_height-new_size[1])//2))  # W*H [0,255]
			assert new_target.size == (fine_width,fine_height)
			# target = self.transform(new_target)		 # 1*256*256  1*H*W ToTensor
			new_target.save(osp.join(layout_save_dir, img_name))

if __name__ == '__main__':
	main()