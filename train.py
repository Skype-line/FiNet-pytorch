import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import itertools
from PIL import Image
import argparse
import os
import os.path as osp
import time
import datetime
import sys
from tensorboardX import SummaryWriter

from Fi_dataset import FiDataset, FiDataLoader
from models import Create_nets, VGGLoss, KL_loss, GANLoss, _freeze, _unfreeze, cal_gradient_penalty
from utils import *

def train_stage1(opt, train_loader, netG, netEI, netEC, netD, board):
	netG.train()
	netEI.train()
	netEC.train()
	netD.train() # unuse

	img_save_dir = osp.join(opt.exp_name, opt.img_result_dir, opt.stage, 'train')
	if not osp.exists(img_save_dir):
		os.makedirs(img_save_dir)

	# criterion
	criterion_seg = nn.CrossEntropyLoss()

	# optimizer
	optimizer_EG = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, netG.parameters()), 
													filter(lambda p: p.requires_grad, netEI.parameters()), 
													filter(lambda p: p.requires_grad, netEC.parameters())), 
															lr=opt.lr, betas=(0.5, 0.999))

	# optimizer = torch.optim.Adam([
	# 	{'params':netG.parameters()},
	# 	{'params':netEI.parameters()},
	# 	{'params':netEC.parameters()}],
	# 	lr=opt.lr, betas=(0.5,0.999) )


	prev_time = time.time()
	for step in range(opt.start_step, opt.train_step):
		inputs = train_loader.next_batch()

		img_name = inputs['img_name']
		xs = inputs['target'].cuda()											# 16*1*128*128
		xc = inputs['context_garments'].cuda()									# 16*12*128*128
		ps = inputs['pp_represent'].cuda()										# 16*19*256*256
		mask = inputs['mask'].cuda()
		mask_info = inputs['mask_info']										
		sc = inputs['masked_context'].cuda()									# 16*8*256*256 [0,1]
		gt = inputs['ground_truth'].cuda()  # groud truth 					# 16*256*256
		# layout_gt = inputs['layout'].cuda()	  # for visualization				# 16*3*256*256!!!!!!!!!!!!!!!!!!!!!!!
		target_type = inputs['target_type']


		_, _, distribution_I, distribution_norm = netEI(xs)
		_, _, distribution_C, _ = netEC(xc)
		zi = distribution_I.rsample()
		# zc = distribution_C.rsample()

		sc_input = (sc-0.5)/0.5 #[-1,1]
		raw_output, post_output = netG(ps, sc_input, zi, z_norm=None, mask=mask)  # raw output: without logsoftmax 32*8*256*256
		
		layout = post_output.detach() * (1-mask) + sc * mask
		
		# ------------------------
		# Train Generator/Encoder
		# ------------------------

		optimizer_EG.zero_grad()

		# KL loss
		G_kl_loss1 = torch.distributions.kl_divergence(distribution_norm, distribution_I).mean()* opt.lambda_kl * opt.output_scale
		G_kl_loss2 = torch.distributions.kl_divergence(distribution_I, distribution_C).mean()* opt.lambda_kl * opt.output_scale
		
		G_kl_loss = G_kl_loss1 + G_kl_loss2 
		
		# segmentation loss
		G_seg_loss = criterion_seg(raw_output, gt)
		
		# Total loss
		G_loss = G_kl_loss + G_seg_loss

		G_loss.backward()
		optimizer_EG.step()

		# ----------------
		# Visulization
		# ----------------
		
		if (step+1) % opt.display_count == 0:
			# board_add_images(board, 'combine', visuals, step+1)

			board.add_scalars('loss', {'Total loss':G_loss.item(),
									 	'Seg loss':G_seg_loss.item(),
									 	'KL loss':G_kl_loss.item()}, step+1)
			board.add_scalar('Total loss',G_loss.item(),step+1)
			board.add_scalar('Seg loss',G_seg_loss.item(),step+1)
			board.add_scalar('KL loss',G_kl_loss.item(),step+1)
			time_left = datetime.timedelta(seconds = (opt.train_step-opt.start_step) * (time.time()-prev_time) / (step-opt.start_step))
			prev_time = time.time()

			sys.stdout.write("\r[Step %d/%d][Seg loss:%f, KL loss:%f, Total loss:%f]ETA: %s" %
				(step+1, opt.train_step, G_seg_loss.data.cpu(),G_kl_loss.data.cpu(), G_loss.data.cpu(), time_left))

		if (step+1) % opt.save_count == 0:
			for i in range(layout.shape[0]):
				layout_c8 = layout[i]
				layout_c1 = seg8_to_seg1(layout_c8)				# 256*256 save!
				layout_c3 = decode_labels(layout_c1)			# 256*256*3 visualization
				layout_show = Image.fromarray(layout_c3)
				# layout_show.save(osp.join(img_save_dir, target_type[i]+str(step).zfill(5)+'_'+img_name[i]))	
				layout_show.save(osp.join(img_save_dir, str(step).zfill(5)+'_'+img_name[i]))	
			save_checkpoint(netG, osp.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netG', 'step_%06d.pth' % (step+1)))
			save_checkpoint(netEI, osp.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netEI', 'step_%06d.pth' % (step+1)))
			save_checkpoint(netEC, osp.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netEC', 'step_%06d.pth' % (step+1)))
			save_checkpoint(netD,  osp.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netD',  'step_%06d.pth' % (step+1)))


def train_stage2(opt, train_loader, netG, netEI, netEC, netD, board):
	netG.train()
	netEI.train()
	netEC.train()
	netD.train()

	img_save_dir = osp.join(opt.exp_name, opt.img_result_dir, opt.stage, 'train')
	if not osp.exists(img_save_dir):
		os.makedirs(img_save_dir)

	# criterion
	criterionL1 = nn.L1Loss()
	criterionL2 = nn.MSELoss()
	criterionVGG = VGGLoss()
	criterion_GAN = GANLoss(opt.gan_mode)

	# optimizer
	# optimizer_EG = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, netG.parameters()),
	# 			                        			filter(lambda p: p.requires_grad, netEI.parameters()),
	# 			                        			filter(lambda p: p.requires_grad, netEC.parameters())), 
	# 														lr=opt.lr, betas=(0.0, 0.999))
	# optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, netD.parameters())),
	# 														lr=opt.lr, betas=(0.0,0.999))
	optimizer_EG = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, netG.parameters()), 
													filter(lambda p: p.requires_grad, netEI.parameters()), 
													filter(lambda p: p.requires_grad, netEC.parameters())), 
															lr=opt.lr, betas=(0.5, 0.999))
	optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=opt.lr, betas=(0.5,0.999))

	prev_time = time.time()
	for step in range(opt.start_step, opt.train_step):
		inputs = train_loader.next_batch()

		img_name = inputs['img_name']
		xs = inputs['target'].cuda()											# 16*3*128*128
		xc = inputs['context_garments'].cuda()									# 16*12*128*128
		# head = inputs['head'].cuda()          # for visualization				# 16*1*256*256
		# pose = inputs['pose_heatmap'].cuda()  # for visualization				# 16*18*256*256
		ps = inputs['pp_represent'].cuda()										# 16*11*256*256
		mask = inputs['mask'].cuda()											# 16*1*256*256
		mask_info = inputs['mask_info']										
		sc = inputs['masked_context'].cuda()									# 16*3*256*256 [-1,1]
		gt = inputs['ground_truth'].cuda()  # groud truth  : full img			# 16*3*256*256
		target_type = inputs['target_type']
		
		scale_gt = scale_pyramid(gt, opt.output_scale)
		scale_mask = scale_pyramid(mask, opt.output_scale)

		_, _, distribution_I, distribution_norm = netEI(xs)
		_, _, distribution_C, _ = netEC(xc)
		zi = distribution_I.rsample()
		zn = distribution_norm.rsample()
		# zc = distribution_C.rsample()

		results = netG(ps, sc, zi, zn, mask)  # tanh 32*3*256*256 !!
		Img_rec = []
		Img_norm = []
		for result in results:
			img_rec, img_norm = result.chunk(2)
			Img_rec.append(img_rec)
			Img_norm.append(img_norm)
		# raw_output = netG(mu1, ps, sc)  # tanh 16*3*256*256
		# masked_layout = select_masked_region(raw_output, mask_info)
		# masked_layout = raw_output * (1 - mask)
		img_out = Img_rec[-1].detach() * (1-mask) + sc * mask

		# --------------------
		# Train Discriminator
		# --------------------

		optimizer_D.zero_grad()
		_unfreeze(netD)


		# ## conditional discriminator##
		# # Real loss
		# D_real = netD(gt,z=zi.detach())
		# D_real_loss = criterion_GAN(D_real, True, True)
		# # Fake loss
		# D_fake_rec = netD(Img_rec[-1].detach(),z=zi.detach())
		# D_fake_loss1 = criterion_GAN(D_fake_rec, False, True)
		# D_fake_norm = netD(Img_norm[-1].detach(),z=zi.detach())
		# D_fake_loss2 = criterion_GAN(D_fake_norm, False, True)

		# Real loss
		D_real = netD(gt)
		D_real_loss = criterion_GAN(D_real, True, True)
		# Fake loss
		D_fake_rec = netD(Img_rec[-1].detach())
		D_fake_loss1 = criterion_GAN(D_fake_rec, False, True)
		D_fake_norm = netD(Img_norm[-1].detach())
		D_fake_loss2 = criterion_GAN(D_fake_norm, False, True)

		# Total loss
		D_loss = (D_real_loss + D_fake_loss1 + D_fake_loss2) * 0.5
		# gradient penalty for wgan-gp
		if opt.gan_mode == 'wgangp':
			gradient_penalty, gradients = cal_gradient_penalty(netD, real, fake.detach())
			D_loss += gradient_penalty

		D_loss.backward()
		optimizer_D.step()

		# ------------------------
		# Train Generator/Encoder
		# ------------------------

		optimizer_EG.zero_grad()
		_freeze(netD)


		# ## conditional discriminator##
		# # GAN loss
		# D_fake_norm = netD(Img_norm[-1],z=zi)
		# G_ad_loss1 = criterion_GAN(D_fake_norm, True, False) * opt.lambda_ad
		
		# D_fake_rec = netD(Img_rec[-1],z=zi)
		# D_real = netD(gt,z=zi)
		# G_ad_loss2 = criterionL2(D_fake_rec, D_real) * opt.lambda_ad


		# GAN loss
		D_fake_norm = netD(Img_norm[-1])
		G_ad_loss1 = criterion_GAN(D_fake_norm, True, False) * opt.lambda_ad
		
		D_fake_rec = netD(Img_rec[-1])
		D_real = netD(gt)
		G_ad_loss2 = criterionL2(D_fake_rec, D_real) * opt.lambda_ad
		
		G_ad_loss = G_ad_loss1 + G_ad_loss2
		# KL loss
		G_kl_loss1 = torch.distributions.kl_divergence(distribution_norm, distribution_I).mean()* opt.lambda_kl * opt.output_scale
		G_kl_loss2 = torch.distributions.kl_divergence(distribution_I, distribution_C).mean()* opt.lambda_kl * opt.output_scale
		
		G_kl_loss = G_kl_loss1 + G_kl_loss2 

		# # vgg loss
		# # # G_rec_loss, G_perceptual_loss, G_style_loss = criterionVGG(masked_layout, sc)
		# # G_app_loss, G_perceptual_loss, G_style_loss = criterionVGG(Img_rec[-1], gt)
		# G_app_loss, G_perceptual_loss, G_style_loss = criterionVGG(Img_rec[-1], gt, opt.lambda_rec)
		
		# calculate l1 loss ofr multi-scale outputs
		G_app_loss1, G_app_loss2 = 0, 0
		for i, (img_rec_i, img_norm_i, img_real_i, mask_i) in enumerate(zip(Img_rec, Img_norm, scale_gt, scale_mask)):
			G_app_loss1 += criterionL1(img_norm_i*mask_i, img_real_i*mask_i)* opt.lambda_rec
			G_app_loss2 += criterionL1(img_rec_i, img_real_i)* opt.lambda_rec

		G_app_loss = G_app_loss1 + G_app_loss2
		
		# Total loss
		G_loss = G_ad_loss + G_kl_loss + G_app_loss

		G_loss.backward()
		optimizer_EG.step()


		# ----------------
		# Visulization
		# ----------------

		if (step+1) % opt.display_count == 0:
			# board_add_images(board, 'combine', visuals, step+1)
			board.add_scalars('Total loss', {'D loss':D_loss.item(),
									   		 'G loss':G_loss.item()}, step+1)
			board.add_scalars('D loss', {'GAN real':D_real_loss.item(),
									   	'GAN fake1':D_fake_loss1.item(),
									   	'GAN fake2':D_fake_loss2.item()}, step+1)
			board.add_scalars('G loss', {'GAN loss':G_ad_loss.item(),
									   	'KL loss':G_kl_loss.item(),
									   	'app loss':G_app_loss.item()}, step+1)
			board.add_scalars('G_GAN loss', {'norm':G_ad_loss1.item(),
									   		 'rec':G_ad_loss2.item()}, step+1)
			board.add_scalars('G_KL loss', {'EI/norm':G_kl_loss1.item(),
									   		'EI/EC':G_kl_loss2.item()}, step+1)
			board.add_scalars('G_app loss', {'norm':G_perceptual_loss.item(),
									   		 'rec':G_style_loss.item()}, step+1)
			# board.add_scalars('G_app loss', {'norm':G_app_loss1.item(),
			# 						   		 'rec':G_app_loss2.item()}, step+1)
			board.add_scalar('D_GAN loss',D_loss.item(),step+1)
			board.add_scalar('G_GAN loss',G_ad_loss.item(),step+1)
			# board.add_scalar('Perceptual loss',G_perceptual_loss.item(),step+1)
			# board.add_scalar('Style loss',G_style_loss.item(),step+1)
			# board.add_scalar('KL loss',G_kl_loss.item(),step+1)
			time_left = datetime.timedelta(seconds = (opt.train_step-opt.start_step) * (time.time()-prev_time) / (step-opt.start_step))
			prev_time = time.time()

			sys.stdout.write("\r[Step %d/%d][D loss:%f, G loss:%f, G_ad loss:%f, G_app loss:%f, G_KL loss:%f]ETA: %s" %
				(step+1, opt.train_step, D_loss.data.cpu(), G_loss.data.cpu(), G_ad_loss.data.cpu(), G_app_loss.data.cpu(), G_kl_loss.data.cpu(), time_left))

		if (step+1) % opt.save_count == 0:
			for i in range(img_out.shape[0]):
				layout = img_out[i]
				layout_show = transforms.ToPILImage()((layout.cpu()+1)/2.0)
				layout_show.save(osp.join(img_save_dir, str(step).zfill(5)+'_'+img_name[i]))	
				# layout_show.save(osp.join(img_save_dir, target_type[i]+str(step).zfill(5)+'_'+img_name[i]))	
			save_checkpoint(netG,  osp.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netG',  'step_%06d.pth' % (step+1)))
			save_checkpoint(netEI, osp.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netEI', 'step_%06d.pth' % (step+1)))
			save_checkpoint(netEC, osp.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netEC', 'step_%06d.pth' % (step+1)))
			save_checkpoint(netD,  osp.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netD',  'step_%06d.pth' % (step+1)))


	

def main():
	opt = Options().parse()
	print("Start to train stage: %s!" % opt.stage)

	# create dataset
	train_dataset = FiDataset(opt)

	#create dataloader
	train_loader = FiDataLoader(opt, train_dataset)

	#visualization
	# if not osp.exists(opt.tensorboard_dir):
	# 	os.makedirs(opt.tensorboard_dir)
	board = SummaryWriter(log_dir = osp.join(opt.exp_name, opt.tensorboard_dir, opt.stage+'_'+opt.mode))

	# create model & train & save the final checkpoint
	netG, netEI, netEC, netD = Create_nets(opt)
	if opt.stage == 'stage1':
		train_stage1(opt, train_loader, netG, netEI, netEC, netD, board)
	elif opt.stage == 'stage2':
		train_stage2(opt, train_loader, netG, netEI, netEC, netD, board)
	else:
		raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
	save_checkpoint(netG, osp.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netG', opt.stage+'_G_final.pth'))
	save_checkpoint(netEI, osp.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netEI', opt.stage+'_EI_final.pth'))
	save_checkpoint(netEC, osp.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netEC', opt.stage+'_EC_final.pth'))
	save_checkpoint(netD, osp.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netD', opt.stage+'_D_final.pth'))
	
	print('Finished training %s!' % opt.stage)

if __name__ == "__main__":
	main()