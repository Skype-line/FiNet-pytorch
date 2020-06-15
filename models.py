import torch
import torch.nn as nn
from torch.nn import init, Parameter
import torch.nn.functional as F
from torch.optim import lr_scheduler

from torchvision import models
from utils import load_checkpoint, seg1_to_seg8, scale_img

import os
import functools

####################################################################################################
# Network Initialization
####################################################################################################

def init_weights(net, init_type='normal', gain=0.02):
    """Get different initial method for the network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def print_networks(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	# print(net)
	print('Total number of parameters of model %s at %s: %6fM' % (net.name, net.stage, num_params/1000000))


####################################################################################################
# Define Loss: kl loss/ vgg loss(perceptual loss + style loss) / GAN loss
####################################################################################################

def KL_loss(mu1, logvar1, mu2, logvar2):
	loss = torch.sum(0.5 * ((logvar2 - logvar1 + 
					(torch.exp(logvar1) + (mu1-mu2).pow(2)) / torch.exp(logvar2) -1)))
	return loss

class Vgg19(nn.Module):
	def __init__(self, requires_grad=False):
		super(Vgg19, self).__init__()
		vgg_pretrained_features = models.vgg19(pretrained=True).features
		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		self.slice5 = torch.nn.Sequential()
		for x in range(2):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(2, 7):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(7, 12):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(12, 21):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for x in range(21, 30):
			self.slice5.add_module(str(x), vgg_pretrained_features[x])
		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, x):
		h_relu1 = self.slice1(x)
		h_relu2 = self.slice2(h_relu1)
		h_relu3 = self.slice3(h_relu2)
		h_relu4 = self.slice4(h_relu3)
		h_relu5 = self.slice5(h_relu4)
		out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
		return out

class GramMatrix(nn.Module):
	def forward(self, x):
		b,c,h,w = x.size()
		x = x.view(b,c,h*w)
		x = torch.bmm(x, x.transpose(1,2))
		x.div_(h*w)
		return x

class VGGLoss(nn.Module):
	def __init__(self, layids = None):
		super(VGGLoss, self).__init__()
		self.vgg = Vgg19()
		self.vgg.cuda()
		self.criterion = nn.L1Loss()
		self.perceptual_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  # cp_vton
		# self.perceptual_weights = [1.0/5.3*2.5, 1.0/2.7/1.2, 1.0/1.35/2.3, 1.0/0.67/8.2, 1.0/0.16] # viton
		self.style_weights = [1.0/1.6, 1.0/2.3, 1.0/1.8, 1.0/2.8, 1.0*10/0.8]
		self.layids = layids

	def forward(self, x, y, lambda_rec):
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)
		perceptual_loss = self.criterion(x,y)
		style_loss = 0
		if self.layids is None:
			self.layids = list(range(len(x_vgg)))
		for i in self.layids:
			# loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
			perceptual_loss += self.perceptual_weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()) * 3 / 5.0 / 128.0
			style_loss += self.style_weights[i] * self.criterion(GramMatrix()(x_vgg[i]),GramMatrix()(y_vgg[i].detach()))/ 5.0
		perceptual_loss = perceptual_loss * lambda_rec
		style_loss = style_loss * lambda_rec
		loss = perceptual_loss + style_loss
		return loss, perceptual_loss, style_loss


# adversarial loss for different gan mode

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, target_is_real, is_disc=False):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            labels = (self.real_label if target_is_real else self.fake_label).expand_as(prediction).type_as(prediction)
            loss = self.loss(prediction, labels)
        elif self.gan_mode in ['hinge', 'wgangp']:
            if is_disc:
                if target_is_real:
                    prediction = -prediction
                if self.gan_mode == 'hinge':
                    loss = self.loss(1 + prediction).mean()
                elif self.gan_mode == 'wgangp':
                    loss = prediction.mean()
            else:
                loss = -prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.type_as(real_data)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).type_as(real_data),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


####################################################################################################
# Define fundamental methods
####################################################################################################
def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False

def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def conv3x3(input_nc, output_nc, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=stride,
					padding=1, bias=False)

def conv1x1(input_nc, output_nc, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=stride, bias=False)


# spectral normalization layer to decouple the magnitude of a weight tensor

def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module

def l2normalize(v, eps=1e-12):
    	return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    """
    spectral normalization
    code and idea originally from Takeru Miyato's work 'Spectral Normalization for GAN'
    https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+1+opt.iter_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(norm_type='batch'):
    """Get the normalization layer for the networks"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, momentum=0.1, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class Auto_Attn(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(Auto_Attn, self).__init__()

        self.query_conv = conv1x1(input_nc, input_nc // 4)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.model = ResBlock(int(input_nc*2), input_nc, input_nc, norm_layer=norm_layer, use_spect=True)

    def forward(self, x, pre=None, mask=None):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H)  # B X (N)X C
        proj_key = proj_query  # B X C x (N)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x

        if type(pre) != type(None):
            # using long distance attention layer to copy information from valid regions
            context_flow = torch.bmm(pre.view(B, -1, W*H), attention.permute(0, 2, 1)).view(B, -1, W, H)
            context_flow = self.alpha * (1-mask) * context_flow + (mask) * pre
            out = self.model(torch.cat([out, context_flow], dim=1))

        return out, attention


class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 sample_type='none', use_spect=False):
        super(ResBlock, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc
        self.sample = True
        if sample_type == 'none':
            self.sample = False
        elif sample_type == 'up':
            output_nc = output_nc * 4
            self.pool = nn.PixelShuffle(upscale_factor=2)
        elif sample_type == 'down':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('sample type [%s] is not found' % sample_type)

        self.conv1 = spectral_norm(conv3x3(input_nc, hidden_nc), use_spect)
        self.conv2 = spectral_norm(conv3x3(hidden_nc, output_nc), use_spect)
        self.bypass = spectral_norm(conv1x1(input_nc, output_nc), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.conv1, nonlinearity, self.conv2)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv1, norm_layer(hidden_nc), nonlinearity, self.conv2)

        self.shortcut = nn.Sequential(self.bypass)

    def forward(self, x):
        if self.sample:
            out = self.pool(self.model(x)) + self.pool(self.shortcut(x))
        else:
            out = self.model(x) + self.shortcut(x)

        return out


class ResBlockLayer1(nn.Module):
    """
    Define an Encoder block for the first layer of the discriminator and representation network
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(), use_spect=False):
        super(ResBlockLayer1, self).__init__()

        self.conv1 = spectral_norm(conv3x3(input_nc, output_nc), use_spect)
        self.conv2 = spectral_norm(conv3x3(output_nc, output_nc), use_spect)
        self.bypass = spectral_norm(conv1x1(input_nc, output_nc), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(self.conv1, nonlinearity, self.conv2, nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.model = nn.Sequential(self.conv1, norm_layer(output_nc), nonlinearity, self.conv2, nn.AvgPool2d(kernel_size=2, stride=2))

        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), self.bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)

        return out


class ResBlockDecoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(), use_spect=False):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc

        self.conv1 = spectral_norm(conv3x3(input_nc, hidden_nc), use_spect)
        self.conv2 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        self.bypass = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.conv1, nonlinearity, self.conv2)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv1, norm_layer(hidden_nc), nonlinearity, self.conv2)

        self.shortcut = nn.Sequential(self.bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)

        return out


class Output(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, stage, input_nc, output_nc, kernel_size=3, nonlinearity= nn.LeakyReLU(), use_spect=False):
        super(Output, self).__init__()

        self.conv1 = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size, bias=True), use_spect)

        if stage == 'stage1':
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1)
        else:
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1, nn.Tanh())

    def forward(self, x):
        out = self.model(x)

        return out
####################################################################################################
# Define Discriminator
####################################################################################################

class Discriminator(nn.Module):
    """
    Stage2
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, stage, name='discriminator', input_nc=3, ndf=32, img_f=128, layers=5, norm='none', activation='LeakyReLU', use_spect=True, use_attn=True):
        super(Discriminator, self).__init__()
        self.name = name
        self.stage = stage
        
        self.layers = layers
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = ResBlockLayer1(input_nc, ndf, norm_layer, nonlinearity, use_spect)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            # self-attention
            if i == 2 and use_attn:
                attn = Auto_Attn(ndf * mult_prev, norm_layer)
                setattr(self, 'attn' + str(i), attn)
            block = ResBlock(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, 'down', use_spect)
            setattr(self, 'encoder' + str(i), block)

        self.block1 = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'none', use_spect)
        self.conv = SpectralNorm(nn.Conv2d(ndf * mult, 1, 3, bias=False))

    def forward(self, x, z=None):
        # conditional GAN
        if type(z) !=type(None):
            x = torch.cat([x,z.repeat(1,1,x.size(2)//z.size(2),x.size(3)//z.size(3))], dim=1)
        out = self.block0(x)
        for i in range(self.layers - 1):
            if i == 2 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.block1(out)
        out = self.conv(self.nonlinearity(out))
        return out

   
####################################################################################################
# Define Encoder
####################################################################################################

class Encoder(nn.Module):  
    def __init__(self, stage, name, ngf=32, z_nc=8, img_f=128, layers=5, norm='none', activation='LeakyReLU', use_spect=True):
        super(Encoder, self).__init__()
        self.name = name
        self.stage = stage
        self.layers = layers
        self.z_nc = z_nc

        if name == 'contextual_encoder':
            input_nc = 12
        elif name == 'input_encoder':
            if stage == 'stage1':
                input_nc = 1
            else:
                input_nc = 3
        elif name == 'generator':
            if stage == 'stage1':
                input_nc = 27 + z_nc
            else:
                input_nc = 14 + z_nc

        norm_layer = get_norm_layer(norm_type=norm)
        
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # encoder part
        self.block0 = ResBlockLayer1(input_nc, ngf, norm_layer, nonlinearity, use_spect)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect)
            setattr(self, 'encoder' + str(i), block)

        self.distribution = ResBlock(ngf * mult, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect)

    def forward(self, x):
        
        # encoder part
        out = self.block0(x)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        if self.name == 'generator':
            return feature
        else:
            o = self.distribution(out)
            mu, std = torch.split(o, self.z_nc, dim=1)
            distribution_e = torch.distributions.Normal(mu, F.softplus(std))
            distribution_norm = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            return mu, std, distribution_e, distribution_norm


####################################################################################################
# Define Generator
####################################################################################################

class Generator(nn.Module):
    def __init__(self, stage, output_scale, name='generator', ngf=32, z_nc=8, img_f=128, layers=5, norm='instance', activation='LeakyReLU', use_spect=True, use_attn=True):
        super(Generator, self).__init__()
        self.name = name
        self.stage = stage
        self.output_scale = output_scale
        self.layers = layers
        self.use_attn = use_attn

        if stage == 'stage1':
            output_nc = 8
        else:
            output_nc = 3

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # encoder part
        self.block0 = Encoder(stage, name, ngf, z_nc, img_f, layers, norm='none', activation='LeakyReLU', use_spect=use_spect)

        # decoder part
        mult = min(2 ** (layers-1), img_f // ngf)
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers - i - 1), img_f // ngf)
            if i > layers - output_scale:
                # upconv = ResBlock(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', use_spect)
                upconv = ResBlockDecoder(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect)
            else:
                # upconv = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', use_spect)
                upconv = ResBlockDecoder(ngf * mult_prev , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect)
            setattr(self, 'decoder' + str(i), upconv)
            # output part
            if i > layers - output_scale - 1:
                outconv = Output(self.stage, ngf * mult, output_nc, 3, nonlinearity, use_spect)
                setattr(self, 'out' + str(i), outconv)
            # short+long term attention part
            if i == 1 and use_attn:
                attn = Auto_Attn(ngf*mult, None)
                setattr(self, 'attn' + str(i), attn)

    def forward(self, p, c , zi, z_norm=None, mask=None):
        x = torch.cat([p,c], dim=1)
        if type(z_norm) != type(None):
            x = torch.cat([x,x], dim=0)
            z = torch.cat([zi,z_norm], dim=0)
        else:
            z = zi
        g_input = torch.cat([x,z.repeat(1,1,x.size(2)//z.size(2),x.size(3)//z.size(3))], dim=1)
        f = self.block0(g_input)

        f_e = f[2]
        out = f[-1]
        results= []
        attn = 0
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
            if i == 1 and self.use_attn:
                # auto attention
                model = getattr(self, 'attn' + str(i))
                scaled_mask = scale_img(mask, size=[f_e.size(2), f_e.size(3)])
                if type(z_norm) != type(None):
                    scaled_mask = torch.cat([scaled_mask,scaled_mask], dim=0)
                out, attn = model(out, f_e, scaled_mask)
            if i > self.layers - self.output_scale - 1:
                model = getattr(self, 'out' + str(i))
                output = model(out)
                results.append(output)
                out = torch.cat([out, output], dim=1)

        if self.stage == 'stage1':
            raw_output = results[-1]
            post_output = seg1_to_seg8(torch.argmax(raw_output, dim=1)).float()  # [2*N,8,256,256]
            return raw_output, post_output
        else:
            return results


####################################################################################################
# Initialize all nets
####################################################################################################

def Create_nets(opt):
	netG  = Generator(stage=opt.stage, output_scale=opt.output_scale, use_attn=True, use_spect=True)
	netEI = Encoder(stage=opt.stage, name='input_encoder', use_spect=True)
	netEC = Encoder(stage=opt.stage, name='contextual_encoder', use_spect=True)
	netD  = Discriminator(stage=opt.stage, input_nc=3, use_attn=True, use_spect=True)
	# netD  = Discriminator(stage=opt.stage, input_nc=11, use_attn=True, use_spect=True)
	# netEI = define_E(1, 8, 64, norm='instance', nl='lrelu', vaeLike=True, stage=opt.stage, name='input_encoder')
	# netEC = define_E(12, 8, 64, norm='instance', nl='lrelu', vaeLike=True, stage=opt.stage, name='contextual_encoder')

	if torch.cuda.is_available():
		netG.cuda()
		netEI.cuda()
		netEC.cuda()
		netD.cuda()

	if (opt.start_step != 0) & (opt.mode == 'train'):
		load_checkpoint(netG, os.path.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netG',  'step_%06d.pth' % opt.start_step))
		load_checkpoint(netEI,os.path.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netEI', 'step_%06d.pth' % opt.start_step))
		load_checkpoint(netEC,os.path.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netEC', 'step_%06d.pth' % opt.start_step))
		load_checkpoint(netD, os.path.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netD',  'step_%06d.pth' % opt.start_step))
	elif (opt.mode == 'test') | (opt.mode == 'exp'):
		load_checkpoint(netG, os.path.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netG',  opt.stage+'_G_final.pth'))
		load_checkpoint(netEI,os.path.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netEI', opt.stage+'_EI_final.pth'))
		load_checkpoint(netEC,os.path.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netEC', opt.stage+'_EC_final.pth'))
		load_checkpoint(netD, os.path.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netD',  opt.stage+'_D_final.pth'))
		# load_checkpoint(netG, os.path.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netG',  'step_060000.pth'))
		# load_checkpoint(netEI,os.path.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netEI', 'step_060000.pth'))
		# load_checkpoint(netEC,os.path.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netEC', 'step_060000.pth'))
		# load_checkpoint(netD, os.path.join(opt.exp_name, opt.checkpoint_dir, opt.stage, 'netD',  'step_060000.pth'))
		# # print_networks(netG)
		# print_networks(netEI)
		# print_networks(netEC)
		# print_networks(netD)
	else:
		print('Initialization start!')
		init_weights(netG, init_type=opt.init_type)
		init_weights(netEI, init_type=opt.init_type)
		init_weights(netEC, init_type=opt.init_type)
		init_weights(netD, init_type=opt.init_type)
		print_networks(netG)
		print_networks(netEI)
		print_networks(netEC)
		print_networks(netD)

	return netG, netEI, netEC, netD





# def meanpoolConv(inplanes, outplanes):
#     sequence = []
#     sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
#     sequence += [nn.Conv2d(inplanes, outplanes,
#                            kernel_size=1, stride=1, padding=0, bias=True)]
#     return nn.Sequential(*sequence)


# def convMeanpool(inplanes, outplanes):
#     sequence = []
#     sequence += [conv3x3(inplanes, outplanes)]
#     sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
#     return nn.Sequential(*sequence)

# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
#         super(BasicBlock, self).__init__()
#         layers = []
#         if norm_layer is not None:
#             layers += [norm_layer(inplanes)]
#         layers += [nl_layer()]
#         layers += [conv3x3(inplanes, inplanes)]
#         if norm_layer is not None:
#             layers += [norm_layer(inplanes)]
#         layers += [nl_layer()]
#         layers += [convMeanpool(inplanes, outplanes)]
#         self.conv = nn.Sequential(*layers)
#         self.shortcut = meanpoolConv(inplanes, outplanes)

#     def forward(self, x):
#         out = self.conv(x) + self.shortcut(x)
#         return out

# class E_ResNet(nn.Module):
#     def __init__(self, input_nc, output_nc, ndf, n_blocks,
#                  norm_layer, nl_layer, vaeLike, stage, name):
#         super(E_ResNet, self).__init__()

#         self.name = name
#         self.stage = stage

#         self.vaeLike = vaeLike
#         max_ndf = 4
#         conv_layers = [
#             nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
#         for n in range(1, n_blocks):
#             input_ndf = ndf * min(max_ndf, n)
#             output_ndf = ndf * min(max_ndf, n + 1)
#             conv_layers += [BasicBlock(input_ndf,
#                                        output_ndf, norm_layer, nl_layer)]
#         conv_layers += [nl_layer(), nn.AvgPool2d(8)]
#         if vaeLike:
#             self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
#             self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
#         else:
#             self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
#         self.conv = nn.Sequential(*conv_layers)

#     def forward(self, x):
#         x_conv = self.conv(x)
#         conv_flat = x_conv.view(x.size(0), -1)
#         output = self.fc(conv_flat)
#         if self.vaeLike:
#             outputVar = self.fcVar(conv_flat)
#             return output, outputVar
#         else:
#             return output
#         return output

# def define_E(input_nc, output_nc, ndf, norm, nl, vaeLike,stage, name):
# 	norm_layer = get_norm_layer(norm_type=norm)
# 	nl_layer = get_non_linearity(layer_type=nl)
# 	net = E_ResNet(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer,
# 				nl_layer=nl_layer, vaeLike=vaeLike, stage=stage, name=name)
# 	return net



# #####################################################################################################
# # Belows are Original Finet
# ####################################################################################################

# class BasicBlock(nn.Module):
#   def __init__(self, input_nc, output_nc):
#       super(BasicBlock, self).__init__()

#       self.conv_block = nn.Sequential(
#           nn.ReLU(inplace=True),
#           conv3x3(input_nc, output_nc),
#           nn.ReLU(inplace=True),
#           conv3x3(output_nc, output_nc) 
#       )

#   def forward(self, x):
#       identity = x
#       out = identity + self.conv_block(x)
#       return out

# class ResidualBlock2(nn.Module):
#   def __init__(self, input_nc, output_nc):
#       super(ResidualBlock2, self).__init__()

#       self.conv = conv1x1(input_nc, output_nc)
#       self.conv_block = nn.Sequential(
#           BasicBlock(output_nc, output_nc),
#           BasicBlock(output_nc, output_nc)
#       )
#       self.relu = nn.ReLU(inplace=True)

#   def forward(self, x):
#       x = self.conv(x)
#       x = self.conv_block(x)
#       out = self.relu(x)
#       return out

# ####################################################################################################
# # Define Encoder
# ####################################################################################################

# class EncoderBlock(nn.Module):
#   def __init__(self, input_nc, output_nc):
#       super(EncoderBlock, self).__init__()

#       layers = [  ResidualBlock2(input_nc, input_nc),
#                   conv3x3(input_nc, output_nc, stride=2),
#                   nn.ReLU(inplace=True) ]
#       self.model = nn.Sequential(*layers)

#   def forward(self, x):
#       out = self.model(x)
#       return out
        
# def sample_z(mu, logvar):
#   # Using reparameterization trick to sample from a gaussian
#   eps = torch.randn(mu.size()).cuda()
#   std = logvar.mul(0.5).exp_()
#   return eps.mul(std).add_(mu)
#   # return mu + torch.exp(logvar / 2) * eps

# class Encoder(nn.Module):
#   def __init__(self, stage, name, ngf=64, n_blocks=6, z_nc=8):
#       super(Encoder, self).__init__()
#       self.name = name
#       self.stage = stage

#       if name == 'contextual_encoder':
#           input_nc = 12
#       elif name == 'input_encoder':
#           if stage == 'stage1':
#               input_nc = 1
#           else:
#               input_nc = 3

#       layers = [  conv3x3(input_nc, ngf, stride=2),
#                   nn.ReLU(inplace=True) ]
#       for i in range(n_blocks):
#           in_ngf = 2**i * ngf if 2**i * ngf < 512 else 512
#           out_ngf = 2**(i+1) * ngf if 2**i * ngf < 512 else 512
#           layers += [EncoderBlock(in_ngf, out_ngf)]

#       self.downmodel = nn.Sequential(*layers)
#       self.fc_mu = nn.Linear(512, z_nc)
#       self.fc_logvar = nn.Linear(512, z_nc)

#   def forward(self, x):
#       x = self.downmodel(x)
#       x = x.view(x.size(0), -1)
#       mu = self.fc_mu(x)
#       logvar = self.fc_logvar(x)
#       z = sample_z(mu, logvar)
#       return z, mu, logvar


# ####################################################################################################
# # Define Generator
# ####################################################################################################

# class UNetDown(nn.Module):
#   def __init__(self, input_nc, hidden_nc, output_nc):
#       super(UNetDown, self).__init__()

#       layers = [  ResidualBlock2(input_nc, hidden_nc),
#                   conv3x3(hidden_nc, output_nc, stride=2),
#                   nn.ReLU(inplace=True) ]
#       self.model = nn.Sequential(*layers)

#   def forward(self, x, z):
#       """ input: x / output: out is the combination of latent code z and featuremap """
#       x = self.model(x)
#       out = torch.cat((x, z), 1)
#       return out

# class UNetUp(nn.Module):
#   def __init__(self, input_nc, output_nc, scale_factor=2):
#       super(UNetUp, self).__init__()

#       layers = [  ResidualBlock2(input_nc, output_nc),
#                   Interpolate(scale_factor)]
#                   # nn.UpsamplingNearest2d(scale_factor=2) ]
#       self.model = nn.Sequential(*layers)

#   def forward(self, x, skip_input):
#       """ input: x / output: out is featuremap before skip connection """
#       x = torch.cat((x, skip_input), 1)
#       out = self.model(x)
#       return out

# class Generator(nn.Module):
#   def __init__(self, stage, name='generator', ngf=64):
#       super(Generator, self).__init__()
#       self.name = name
#       self.stage = stage

#       input_nc = 27 if stage == 'stage1' else 14
#       output_nc = 8 if stage == 'stage1' else 3

#       self.down1 = nn.Sequential(
#           conv3x3(input_nc, 64, stride=2),
#           nn.ReLU(inplace=True))
#       self.down2 = UNetDown(8+64, 64, 128)
#       self.down3 = UNetDown(8+128, 128, 256)
#       self.down4 = UNetDown(8+256, 256, 512)
#       self.down5 = UNetDown(8+512, 512, 512)
#       self.down6 = UNetDown(8+512, 512, 512)
#       self.down7 = UNetDown(8+512, 512, 512)
#       self.down8 = UNetDown(8+512, 512, 512)

#       up1_layers = [  ResidualBlock2(8+512, 512),
#                       Interpolate(scale_factor=2)]
#                       # nn.UpsamplingNearest2d(scale_factor=2)]
#       self.up1 = nn.Sequential(*up1_layers)
#       self.up2 = UNetUp(8+2*512, 512)
#       self.up3 = UNetUp(8+2*512, 512)
#       self.up4 = UNetUp(8+2*512, 512)
#       self.up5 = UNetUp(8+2*512, 256)
#       self.up6 = UNetUp(8+2*256, 128)
#       self.up7 = UNetUp(8+2*128, 64)
#       self.up8 = UNetUp(8+2*64, 32)
#       if stage == 'stage1':
#           self.final = conv3x3(32, output_nc)
#       else:
#           self.final = nn.Sequential(
#                   conv3x3(32, output_nc),
#                   nn.Tanh()) 
         
#   def forward(self, z, p, c):
#       z.unsqueeze_(-1).unsqueeze_(-1)
#       # z = z.detach()
#       x = torch.cat((p, c), 1)
#       d1 = torch.cat((self.down1(x),z.repeat(1,1,128,128)), 1)
#       d2 = self.down2(d1,z.repeat(1,1,64,64))
#       d3 = self.down3(d2,z.repeat(1,1,32,32))
#       d4 = self.down4(d3,z.repeat(1,1,16,16))
#       d5 = self.down5(d4,z.repeat(1,1,8,8))
#       d6 = self.down6(d5,z.repeat(1,1,4,4))
#       d7 = self.down7(d6,z.repeat(1,1,2,2))
#       d8 = self.down8(d7,z)

#       u1 = self.up1(d8)
#       u2 = self.up2(u1, d7)
#       u3 = self.up3(u2, d6)
#       u4 = self.up4(u3, d5)
#       u5 = self.up5(u4, d4)
#       u6 = self.up6(u5, d3)
#       u7 = self.up7(u6, d2)
#       u8 = self.up8(u7, d1)

#       raw_output = self.final(u8)
#       if self.stage == 'stage2':
#           return raw_output
#       else:
#           post_output = seg1_to_seg8(torch.argmax(raw_output, dim=1)).float()  # [N,8,256,256]
#           return raw_output, post_output
