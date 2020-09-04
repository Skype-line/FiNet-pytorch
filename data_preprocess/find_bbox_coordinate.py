# import torch
from PIL import Image
import numpy as np
import os
import os.path as osp
import json
from collections import OrderedDict

def main():
	MODE = ['train','test']
	for mode in MODE:
		imgparse_dir = osp.join(mode,'image-parse')
		bbox_coordiante_dir = osp.join(mode,'bbox_coordinate') # output dir
		
		if not osp.exists(bbox_coordiante_dir):
			os.makedirs(bbox_coordiante_dir)

		img_names = os.listdir(imgparse_dir)
		img_names.sort()

		for item in img_names:
			if not item.endswith('vis.png'):
				im_parse = Image.open(osp.join(imgparse_dir, item))
				parse_array = np.array(im_parse)

				parse_hat = (parse_array == 4).astype(np.float32)
				parse_top = (parse_array == 2).astype(np.float32) + (parse_array == 5).astype(np.float32)
				parse_bottom = (parse_array == 3).astype(np.float32) + (parse_array == 6).astype(np.float32)
				parse_shoes = (parse_array == 7).astype(np.float32)
				parse_items = [parse_hat, parse_top, parse_bottom, parse_shoes]
				item_names = ['hat','top','bottom','shoes']

				Box_Coordinates = OrderedDict()
				for index, item_name in enumerate(item_names):
					parse_item = parse_items[index]
					x_left, x_right, y_up, y_down = find_resized_bbox(parse_item)
					Box_Coordinates[item_name] = {'x_left':x_left, 'x_right':x_right, 'y_up':y_up, 'y_down':y_down}

				with open(osp.join(bbox_coordiante_dir, item.replace('.png','.json')), 'w') as f:
					json.dump(Box_Coordinates,f)
				 
def find_resized_bbox(parse_item, resize_rate = 1.1):
	try:
		x_left = np.nonzero(np.sum(parse_item,axis=0))[0][0]
		x_right = np.nonzero(np.sum(parse_item,axis=0))[0][-1]
		y_up = np.nonzero(np.sum(parse_item,axis=1))[0][0]
		y_down = np.nonzero(np.sum(parse_item,axis=1))[0][-1]
		w = x_right - x_left
		h = y_down - y_up
		x_mid = int((x_left + x_right)/2)
		y_mid = int((y_down + y_up)/2)

		x_left =  x_mid - int(w*resize_rate/2)
		x_right = x_mid + int(w*resize_rate/2)
		y_up =  y_mid - int(h/2)
		y_down = y_mid + int(h/2)
		# if w > h:
		# 	x_left =  x_mid - int(w*resize_rate/2)
		# 	x_right = x_mid + int(w*resize_rate/2)
		# 	y_up =  y_mid - int(w*resize_rate/2)
		# 	y_down = y_mid + int(w*resize_rate/2)
		# else:
		# 	x_left =  x_mid - int(h*resize_rate/2)
		# 	x_right = x_mid + int(h*resize_rate/2)
		# 	y_up =  y_mid - int(h*resize_rate/2)
		# 	y_down = y_mid + int(h*resize_rate/2)
	except IndexError:
		x_left, x_right, y_up, y_down = [],[],[],[]

	return x_left, x_right, y_up, y_down

if __name__ == "__main__":
	main()			
