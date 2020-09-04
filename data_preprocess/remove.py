import os
import os.path as osp
import json

Mode = ['train','test']

for mode in Mode:
	pose_dir = osp.join(mode,'pose')
	image_dir = osp.join(mode,'image')
	image_parse_dir = osp.join(mode,'image-parse')
	posenames = os.listdir(pose_dir)
	posenames.sort()

	for posejson in posenames:
		with open(osp.join(pose_dir,posejson)) as f:
			data = json.load(f)
			if len(data["pose_keypoints"])!=54:
				print(osp.join(pose_dir,posejson))
				# print(osp.join(image_dir,posejson[:6]+'.jpg'))
				# print(osp.join(image_parse_dir,posejson[:6]+'.png'))
				os.remove(osp.join(pose_dir,posejson))
				os.remove(osp.join(image_dir,posejson[:6]+'.jpg'))
				os.remove(osp.join(image_parse_dir,posejson[:6]+'.png'))



