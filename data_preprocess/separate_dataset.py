import os
import os.path as osp
import shutil

old_dir = 'temp'
pose_dir = osp.join(old_dir,'pose')
bbox_dir = osp.join(old_dir,'bbox_coordinate')
image_dir = osp.join(old_dir,'image')
image_parse_dir = osp.join(old_dir,'image-parse')
posenames = os.listdir(pose_dir)
posenames.sort()

pose_train_newdir = osp.join('train','pose')
bbox_train_newdir = osp.join('train','bbox_coordinate')
img_train_newdir = osp.join('train','image')
image_parse_train_newdir = osp.join('train','image-parse')

pose_test_newdir = osp.join('test','pose')
bbox_test_newdir = osp.join('test','bbox_coordinate')
img_test_newdir = osp.join('test','image')
image_parse_test_newdir = osp.join('test','image-parse')

for path in [pose_train_newdir,bbox_train_newdir,img_train_newdir,image_parse_train_newdir,pose_test_newdir,bbox_test_newdir,img_test_newdir,image_parse_test_newdir]:
	if not os.path.exists(path):
		os.makedirs(path)
f1 = open("./train.txt", "w")
f2 = open("./test.txt", "w")
for index, posejson in enumerate(posenames):
	if index%8:
		shutil.copy(osp.join(pose_dir,posejson), osp.join(pose_train_newdir,posejson))
		shutil.copy(osp.join(bbox_dir,posejson[:6]+'.json'), osp.join(bbox_train_newdir,posejson[:6]+'.json'))
		shutil.copy(osp.join(image_dir,posejson[:6]+'.jpg'), osp.join(img_train_newdir,posejson[:6]+'.jpg'))
		shutil.copy(osp.join(image_parse_dir,posejson[:6]+'.png'), osp.join(image_parse_train_newdir,posejson[:6]+'.png'))
		f1.write(posejson[:6]+'.jpg'+ "\n")
	else:
		shutil.copy(osp.join(pose_dir,posejson), osp.join(pose_test_newdir,posejson))
		shutil.copy(osp.join(bbox_dir,posejson[:6]+'.json'), osp.join(bbox_test_newdir,posejson[:6]+'.json'))
		shutil.copy(osp.join(image_dir,posejson[:6]+'.jpg'), osp.join(img_test_newdir,posejson[:6]+'.jpg'))
		shutil.copy(osp.join(image_parse_dir,posejson[:6]+'.png'), osp.join(image_parse_test_newdir,posejson[:6]+'.png'))
		f2.write(posejson[:6]+'.jpg'+ "\n")
f1.close()
f2.close()





