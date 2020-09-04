import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import os
import os.path as osp
import numpy as np
from PIL import Image


class Vgg19_F(nn.Module):
	def __init__(self, requires_grad=False):
		super(Vgg19_F, self).__init__()
		self.F = models.vgg19(pretrained=True).features
		self.C = models.vgg19(pretrained=True).classifier[0:6]
		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, x):
		x = self.F(x)
		x = x.view(x.size(0),-1)
		x = self.C(x)
		return x

def main():
	transform = transforms.Compose([  \
				transforms.ToTensor(),   \
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #[-1,1] Tensor
	# VGG_f = torch.nn.Sequential(
	# 				models.vgg19(pretrained=True).features,
	# 				models.vgg19(pretrained=True).classifier[0:6])
	# Vgg19_F = Vgg19_F.eval()
	VGG_f = Vgg19_F().cuda()

	MODE = ['train','test']
	for mode in MODE:
		layout_path = osp.join(mode,'layout_top')
		feature_path = osp.join(mode,'feature_top')

		if not osp.exists(feature_path):
			os.makedirs(feature_path)

		img_names = os.listdir(layout_path)
		img_names.sort()

		for item in img_names:
			layout = Image.open(osp.join(layout_path, item))
			layout = layout.resize((224,224), Image.ANTIALIAS)
			layout = transform(layout).unsqueeze(0).repeat(1,3,1,1).cuda()

			result = VGG_f(layout)
			result_npy = result.data.cpu().numpy()
			np.save(osp.join(feature_path, "%s.npy" % item[:-4]),result_npy)

if __name__ == "__main__":
	main()