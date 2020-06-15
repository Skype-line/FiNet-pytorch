import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
import argparse
import os
import os.path as osp
import time
import datetime
import sys
import cv2
import numpy as np
from tensorboardX import SummaryWriter

from Fi_dataset import FiDataset, FiDataLoader
from models import Create_nets, VGGLoss, KL_loss, GANLoss, _freeze, _unfreeze, cal_gradient_penalty
from utils import *

def test_stage1(opt, test_loader, netG, netEI, netEC, board):
	netG.eval()
	netEI.eval()
	netEC.eval()

	img_save_dir = osp.join(opt.exp_name, opt.img_result_dir, opt.stage, 'exp_hat')
	if not osp.exists(img_save_dir):
		os.makedirs(img_save_dir)

	prev_time = time.time()
	for step, inputs in enumerate(test_loader.data_loader):
		img_name = inputs['img_name']
		# xs = inputs['target'].cuda()											# 16*1*128*128
		xc = inputs['context_garments'].cuda()									# 16*12*128*128
		ps = inputs['pp_represent'].cuda()										# 16*19*256*256
		mask = inputs['mask'].cuda()
		mask_info = inputs['mask_info']										
		sc = inputs['masked_context'].cuda()									# 16*8*256*256 [0,1]
		target_type = inputs['target_type']

		with torch.no_grad():
			# _ , mu1, logvar1 = netEI(xs)
			zc, mu2, logvar2 = netEC(xc)
			# print(mu1)
			# print(logvar1)
			# print(mu2)
			# print(logvar2)
			# zc = sample_z(mu2, logvar2)
			# print(zc)
			# change the values in different dimensions of the learned latent vector, Generated 8 layouts of 1 input
			if (opt.mode == 'exp1'):#|(opt.mode == 'test'):
				dim = 1
				# weight_list = [0, 10, 20, 30, 40, 44, 48, 52, 
							  # 56, 60, 64, 68, 72, 76, 80, 84]  # dim2
				weight_list = [32, 36, 40, 44, 48, 52, 56, 60]
				# weight_list = [-50, -40, -30, -20, -10, 0, 40, 60]
				# weight_list = [-1000, -100, -10, -1, 1, 10, 100, 1000]
				for i in range(mu2.shape[0]//2):
					# batch size = 16, 1 input image -> 8 output, so 1 batch -> 2 input image
					# mu2[i,dim] = mu2[i,dim] * weight_list[i] # mu2[i,x], x represents xth dimension of the latent code
					# mu2[i+mu2.shape[0]//2,dim] = mu2[i+mu2.shape[0]//2,dim] * weight_list[i]
					mu2[i,dim-1] = mu2[i,dim-1] + weight_list[i]*(logvar2[i,dim-1].mul(0.5).exp_()) # mu2[i,x], x represents xth dimension of the latent code
					mu2[i+mu2.shape[0]//2,dim-1] = mu2[i+mu2.shape[0]//2,dim-1] + weight_list[i]*(logvar2[i+mu2.shape[0]//2,dim-1].mul(0.5).exp_())
		# 	# zc = torch.randn(16,8).cuda()
			# print(mu2)
			sc_input = (sc-0.5)/0.5 #[-1,1]
			# _, post_output = netG(mu2, ps, sc_input)  # raw output: without logsoftmax 16*8*256*256
			_, post_output = netG(zc, ps, sc_input)  # raw output: without logsoftmax 16*8*256*256

		if (step+1) % opt.save_count == 0:
			# layout_c8 = paste(opt.stage, post_output, sc, mask_info)	# 16*8*256*256
			# layout_c1 = seg8_to_seg1(layout_c8)				# 16*1*256*256 save!
			# layout_c3 = decode_labels(layout_c1)			# 16*3*256*256 visualization
			# for i in range(layout_c3.shape[0]):
			for i in range(post_output.shape[0]):
				layout_c8 = post_output[i] * mask[i] + sc[i] * (1-mask[i])
				layout_c1 = seg8_to_seg1(layout_c8)				# 256*256 save!
				layout_c3 = decode_labels(layout_c1)			# 256*256*3 visualization
				layout_show = Image.fromarray(layout_c3)
				# layout_show = Image.fromarray(layout_c3[i])
				# layput_save = np.array(layout_c1[i][0].detach().cpu())
				layput_save = np.array(layout_c1.detach().cpu())
				if opt.mode == 'test':
					# layout_show.save(osp.join(img_save_dir, img_name[i][:-4]+'_'+str(step)+str(i)+'.jpg'))
					# layout_show.save(osp.join(img_save_dir, img_name[i][:-4]+target_type[i]+'_'+str(step)+str(i)+'.jpg'))
					layout_show.save(osp.join(img_save_dir, img_name[i]))
					cv2.imwrite('{}/{}.png'.format(img_save_dir, img_name[i][:-4]), layput_save)
				elif opt.mode == 'exp':
					# layout_show.save(osp.join(img_save_dir, img_name[i][:-4]+'_'+str(step)+str(i)+'.jpg'))
					layout_show.save(osp.join(img_save_dir, img_name[i]))
		if (step+1) % opt.display_count == 0:
			time_left = datetime.timedelta(seconds = (len(test_loader.data_loader) * (time.time()-prev_time) / (step+1)))
			prev_time = time.time()

			sys.stdout.write("\r[Step %d/%d]ETA: %s" % (step+1, len(test_loader.data_loader), time_left))


def test_stage2(opt, test_loader, netG, netEI, netEC, netD,  board):
	netG.eval()
	netEI.eval()
	netEC.eval()
	netD.eval()

	img_save_dir = osp.join(opt.exp_name, opt.img_result_dir, opt.stage, opt.mode)
	if not osp.exists(img_save_dir):
		os.makedirs(img_save_dir)

	prev_time = time.time()
	for step, inputs in enumerate(test_loader.data_loader):
		img_name = inputs['img_name']
		# xs = inputs['target'].cuda()											# 16*3*128*128
		xc = inputs['context_garments'].cuda()									# 16*12*128*128
		ps = inputs['pp_represent'].cuda()										# 16*11*256*256
		mask = inputs['mask'].cuda()
		mask_info = inputs['mask_info']										
		sc = inputs['masked_context'].cuda()									# 16*3*256*256 [0,1]
		target_type = inputs['target_type']

		with torch.no_grad():
			# _ , mu1, logvar1 = netEI(xs)
			mu2, std2, distribution_C, _  = netEC(xc)
			zc = distribution_C.sample()
			if (opt.mode == 'exp'):#|(opt.mode == 'test'):
				dim = 1
				# weight_list = [0, 10, 20, 30, 40, 44, 48, 52, 
							  # 56, 60, 64, 68, 72, 76, 80, 84]  # dim2
				# weight_list = [32, 36, 40, 44, 48, 52, 56, 60]
				weight_list = [-50, -40, -30, -20, -10, 0, 40, 60]
				# weight_list = [-1000, -100, -10, -1, 1, 10, 100, 1000]
				for i in range(mu2.shape[0]//2):
					# batch size = 16, 1 input image -> 8 output, so 1 batch -> 2 input image
					# mu2[i,dim] = mu2[i,dim] * weight_list[i] # mu2[i,x], x represents xth dimension of the latent code
					# mu2[i+mu2.shape[0]//2,dim] = mu2[i+mu2.shape[0]//2,dim] * weight_list[i]
					mu2[i,:] = mu2[i,:] + weight_list[i]*std2[i,:] # mu2[i,x], x represents xth dimension of the latent code
					mu2[i+mu2.shape[0]//2,:] = mu2[i+mu2.shape[0]//2,:] + weight_list[i]*std2[i+mu2.shape[0]//2,:]
					# mu2[i,dim-1] = mu2[i,dim-1] + weight_list[i]*std2[i,dim-1] # mu2[i,x], x represents xth dimension of the latent code
					# mu2[i+mu2.shape[0]//2,dim-1] = mu2[i+mu2.shape[0]//2,dim-1] + weight_list[i]*std2[i+mu2.shape[0]//2,dim-1]
		
			# raw_output = netG(mu2, ps, sc)   # raw output16*3*256*256
			results = netG(ps, sc, zc, z_norm=None, mask=mask)  # raw output 16*3*256*256
			# results = netG(ps, sc, mu2, z_norm=None, mask=mask)  # raw output 16*3*256*256
			img_out = results[-1].detach() * (1-mask) + sc * mask

		if (step+1) % opt.save_count == 0:
			# layout = paste(opt.stage, output, sc, mask_info)	# 16*3*256*256
			# for i in range(layout.shape[0]):
			for i in range(img_out.shape[0]):
				layout = img_out[i]
				layout_show = transforms.ToPILImage()((layout.cpu()+1)/2.0)
				if opt.mode == 'test':
					# layout_show.save(osp.join(img_save_dir, img_name[i][:-4]+'_'+str(step)+str(i)+'.jpg'))
					# layout_show.save(osp.join(img_save_dir, target_type[i]+img_name[i][:-4]+'_'+str(step)+str(i)+'.jpg'))
					layout_show.save(osp.join(img_save_dir, img_name[i]))
				elif opt.mode == 'exp':
					layout_show.save(osp.join(img_save_dir, img_name[i][:-4]+'_'+str(step)+str(i)+'.jpg'))

		if (step+1) % opt.display_count == 0:
			time_left = datetime.timedelta(seconds = (len(test_loader.data_loader) * (time.time()-prev_time) / (step+1)))
			prev_time = time.time()

			sys.stdout.write("\r[Step %d/%d]ETA: %s" % (step+1, len(test_loader.data_loader), time_left))



def main():
	opt = Options().parse()
	print("Start to %s stage: %s!" % (opt.mode,opt.stage))
	# create dataset
	test_dataset = FiDataset(opt)

	#create dataloader
	test_loader = FiDataLoader(opt, test_dataset)

	#visualization
	if not osp.exists(opt.tensorboard_dir):
		os.makedirs(opt.tensorboard_dir)
	board = SummaryWriter(log_dir = osp.join(opt.exp_name, opt.tensorboard_dir, opt.stage+'_'+opt.mode))

	# create model & load the final checkpoint & test 
	netG, netEI, netEC, netD= Create_nets(opt)
	if opt.stage == 'stage1':
		with torch.no_grad():
			test_stage1(opt, test_loader, netG, netEI, netEC, board)
	elif opt.stage == 'stage2':
		with torch.no_grad():
			test_stage2(opt, test_loader, netG, netEI, netEC, netD,  board)
	else:
		raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

	print('Finished test %s!' % opt.stage)


if __name__ == "__main__":
	main()