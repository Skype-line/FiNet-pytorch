import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

class Options():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--exp_name', default='Exp0', help='the name of experiment')
		self.parser.add_argument('--stage', default='stage1', help='stage1: shape generation, stage2: appearance generation')
		self.parser.add_argument('--mode', default='train', help='train/test')
		self.parser.add_argument('--target_type', default='top', help='the type of the inpainted region')
		self.parser.add_argument('--gan_mode', default='lsgan', help='type of GAN objective. [vanilla | lsgan | wgangp]')

		self.parser.add_argument("--dataroot", default = "dataset")
		self.parser.add_argument("--data_list", default = "train.txt", help='train.txt/test.txt')
		self.parser.add_argument("--init_type", default = "normal", help='initialization method of netG,E_I,E_C [normal | xavier | kaiming]')
		self.parser.add_argument("--fine_width", type=int, default = 256)
		self.parser.add_argument("--fine_height", type=int, default = 256)
		self.parser.add_argument("--output_scale", type=int, default = 4)
		self.parser.add_argument("--radius", type=int, default = 5, help='radius of the visualized keypoint')
		self.parser.add_argument('--img_result_dir', default='result_images', help='where to save the result images')
		self.parser.add_argument('--checkpoint_dir', default='checkpoints', help='save/load checkpoints infos')
		# self.parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
		self.parser.add_argument('--tensorboard_dir', default='tensorboard', help='save tensorboard infos')
		self.parser.add_argument('--start_step', type=int, default=0, help='start from where last stop')
		self.parser.add_argument('-b', '--batch_size', type=int, default=16)
		self.parser.add_argument('-j', '--workers', type=int, default=1)
		self.parser.add_argument("--display_count", type=int, default = 20)
		self.parser.add_argument("--save_count", type=int, default = 1000)
		self.parser.add_argument("--train_step", type=int, default = 20000)
		self.parser.add_argument('--lr', type=float, default=0.0001, help='fixed learning rate for adam')
		self.parser.add_argument("--lambda_rec", type=float, default = 20, help='weight for image reconstruction loss')
		self.parser.add_argument("--lambda_kl", type=float, default = 20, help='weight for kl divergence loss')
		self.parser.add_argument("--lambda_ad", type=float, default = 1, help='weight for generation loss')
		self.parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
		# discriminate the train and test mode!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def parse(self):
		if not self.initialized:
			self.initialize()
		args = self.parser.parse_args()
		if args.stage == 'stage2':
			args.train_step = 60000
		else:
			args.output_scale = 1
		if args.mode == 'test':
			args.save_count = 1
			args.display_count = 1
			args.data_list = "test.txt"
		elif args.mode == 'exp':
			args.save_count = 1
			args.display_count = 1
			args.data_list = "exp.txt"

		os.makedirs(args.exp_name, exist_ok=True)

		print('----------------options-----------------')
		with open('./%s/%s_%s_args.log' % (args.exp_name, args.mode, args.stage), 'w') as f:
			for k, v in sorted(vars(args).items()):
				print('%s: %s' % (str(k), str(v)))
				f.write('%s: %s \n' % (str(k), str(v)))
		print('------------------End-------------------')
		self.args = args

		return self.args

"""
Combine the parser result of 'Instance-level human parsing via part grouping network'
from 20 classes to 8 classes
"""
label_colours = [(0,0,0),
                # 0 = background = background+glove+socks+scarf = 0+3+8+11
                (255,200,150), (0,150,255), (50,255,0), (255,0,0), (255,100,0),  (0,85,85), (255,150,0)]
                # 1 = head = face+hair+glasses = 13+2+4	
                # 2 = upper body skin = torso+arms = 10+14+15
                # 3 = lower body skin = legs = 16+17
                # 4 = hat = 1
                # 5 = top clothes = upper-cloth+coat = 5+7
                # 6 = bottom clothes = pants+skirt+dress = 9+12+6
                # 7 = shoes = 18+19  										

def decode_labels(mask, num_classes=8):
    """Decode a single segmentation mask.
    
    Args:
      mask: result of inference after taking argmax.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A single RGB image of the same size as the input. 
    """
    h, w = mask.shape
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    img = Image.new('RGB', (w,h))
    pixels = img.load()
    for j_, j in enumerate(mask[:,:]):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_,j_] = label_colours[k]
    outputs = np.array(img)
    return outputs

def seg8_to_seg1(segmentation8):
	"""
	change segmentation from 8 channels to 1 channels for save as jpg
	"""
	segmentation1 = torch.argmax(segmentation8, dim=0)

	return segmentation1

def seg1_to_seg8(segmentation1):
	"""
	change segmentation from 1 channels to 8 channels for model data input
	"""
	# generator input preprocess
	if (type(segmentation1) == np.ndarray) & (len(segmentation1.shape) == 2): 
		segmentation8 = torch.from_numpy(np.concatenate(((segmentation1==0).astype(np.float32)[np.newaxis,:],
														 (segmentation1==1).astype(np.float32)[np.newaxis,:],
														 (segmentation1==2).astype(np.float32)[np.newaxis,:],
														 (segmentation1==3).astype(np.float32)[np.newaxis,:],
														 (segmentation1==4).astype(np.float32)[np.newaxis,:],
														 (segmentation1==5).astype(np.float32)[np.newaxis,:],
														 (segmentation1==6).astype(np.float32)[np.newaxis,:],
														 (segmentation1==7).astype(np.float32)[np.newaxis,:]),axis=0))
	# generator ouput postprocess
	elif torch.is_tensor(segmentation1) & (len(segmentation1.shape) == 3):
		segmentation8 = torch.stack((segmentation1==0,segmentation1==1,segmentation1==2,
									 segmentation1==3,segmentation1==4,segmentation1==5,
									 segmentation1==6,segmentation1==7),dim=1)
	return segmentation8

def scale_img(img, size):
    scaled_img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
    return scaled_img

def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h // ratio
        nw = w // ratio
        scaled_img = scale_img(img, size=[nh, nw])
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse() # small --> large
    return scaled_imgs

# def paste(stage, inpainted_region, masked_img, mask_info):
# 	batch_size = inpainted_region.shape[0]
# 	for i in range(batch_size):
# 		# resize inpainted image from C*255*255 to C*mask_H*mask_W
# 		paste_region = inpainted_region[i]
# 		H, W = int(mask_info[4][i]), int(mask_info[5][i])
# 		y_up, y_down, x_left, x_right = int(mask_info[0][i]),int(mask_info[1][i]),int(mask_info[2][i]),int(mask_info[3][i])
# 		paste_region = F.interpolate(paste_region.float().unsqueeze(0),size=(H,W),mode='bilinear',align_corners=True)[0]
# 		if stage == 'stage1':
# 			paste_region = paste_region.byte() # resize to mask size [0,1] Tensor
# 		# paste the inpainted image onto the masked image
# 		masked_img[i,:,y_up:y_down,x_left:x_right] = paste_region
	
# 	return masked_img

def select_masked_region(generate_layout, mask_info):
	batch_size, H, W = generate_layout.shape[0], generate_layout.shape[2], generate_layout.shape[3]
	masked_layout = torch.zeros(generate_layout.shape).cuda()
	for i in range(batch_size):
		layout = generate_layout[i]
		y_up, y_down, x_left, x_right = int(mask_info[0][i]),int(mask_info[1][i]),int(mask_info[2][i]),int(mask_info[3][i])
		inpainted_region = layout[:,y_up:y_down,x_left:x_right]
		resized_region = F.interpolate(inpainted_region.float().unsqueeze(0),size=(H,W),mode='bilinear',align_corners=True)[0]
		masked_layout[i] = resized_region

	return masked_layout

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("checkpoint %s not found!" % checkpoint_path)
        return
    model.load_state_dict(torch.load(checkpoint_path))
    if torch.cuda.is_available():
        model.cuda()