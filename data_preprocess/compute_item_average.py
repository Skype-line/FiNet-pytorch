import torch
import os.path as osp
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json

fine_height_E = 256 #!!!!
fine_width_E = 256
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

MODE = ['train','test']  #test
for mode in MODE:
	img_names = []
	with open(mode+'.txt', 'r') as f:
		for line in f.readlines():
			img_names.append(line.strip())
	N = len(img_names)

	average_items = {'hat':torch.zeros(3,fine_height_E,fine_width_E),\
					 'top':torch.zeros(3,fine_height_E,fine_width_E),\
					 'bottom':torch.zeros(3,fine_height_E,fine_width_E),\
					 'shoes':torch.zeros(3,fine_height_E,fine_width_E)}
	N_items = {'hat':N, 'top':N, 'bottom':N, 'shoes':N}

	for img_name in img_names:
		im = Image.open(osp.join(mode, 'image', img_name))
		im = transforms.ToTensor()(im)

		parse_name = img_name.replace('.jpg','.png')
		im_parse = Image.open(osp.join(mode, 'image-parse', parse_name))
		parse_array = np.array(im_parse) # H*W

		parse_hat = (parse_array == 4).astype(np.float32)
		parse_top_cloth = (parse_array == 5).astype(np.float32)
		parse_bottom_cloth = (parse_array == 6).astype(np.float32)
		parse_shoes = (parse_array == 7).astype(np.float32)

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
		with open(osp.join(mode, 'bbox_coordinate', bbox_name), 'r') as f:
			box = json.load(f) 
		for item_name in ['hat','top','bottom','shoes']:
			try:
				coor = box[item_name]
				crop_item = im_items[item_name][:,coor['y_up']:coor['y_down'],coor['x_left']:coor['x_right']] #[-1,1]
				crop_item = transforms.ToPILImage()(crop_item)
				ratio = fine_height_E / max(crop_item.size)
				new_size = tuple([int(x*ratio) for x in crop_item.size])
				resized_item = crop_item.resize(new_size, Image.ANTIALIAS)
				new_item = Image.new("RGB",(fine_width_E,fine_height_E),color=(255,255,255))
				new_item.paste(resized_item, ((fine_width_E-new_size[0])//2,
												(fine_height_E-new_size[1])//2))  # W*H [0,255]
				assert new_item.size == (fine_width_E,fine_height_E)
				new_item = transform(new_item) # [-1,1] 3*128*128  3*H*W Tensor
				average_items[item_name] += new_item
			except TypeError:
				N_items[item_name] -= 1
			
	for item_name in ['hat','top','bottom','shoes']:
		average_items[item_name] /= N_items[item_name]
		np.save(osp.join(mode,"%s_%s_mean.npy" % (item_name,mode)),np.array(average_items[item_name]))
		average_show = transforms.ToPILImage()((average_items[item_name]*0.5)+0.5)
		average_show.save(osp.join(mode,"%s_%s_mean.jpg" % (item_name,mode)))
		# print(average_items[item_name])
		# print(N_items[item_name])
		# print(torch.from_numpy(np.load(osp.join(mode,"%s_%s_mean.npy" % (item_name,mode))))==average_items[item_name])

# from IPython import embed; embed()