# coding=utf-8

"""
ResNet.

Reference:
	[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
		Deep Residual Learning for Image Recognition. arXiv:1512.03385

	[2] https://github.com/pytorch/vision.
"""

import torch
import torch.nn as nn

try:
	from torch.hub import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['resnet110_sl','wide_resnetsl50_2','wide_resnetsl16_8']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
						padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
						base_width=64, dilation=1, norm_layer=None):
		super(BasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class Bottleneck(nn.Module):
	# Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
	# while original implementation places the stride at the first 1x1 convolution(self.conv1)
	# according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
	# This variant is also known as ResNet V1.5 and improves accuracy according to
	# https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
					base_width=64, dilation=1, norm_layer=None):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups

		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out
# Main client model
class ResNetMainClient(nn.Module):

	def __init__(self, arch, block, layers, num_classes=1000, zero_init_residual=True,
					groups=1, width_per_group=64, replace_stride_with_dilation=None,
					norm_layer=None, dataset='cifar10', split_factor=1, output_stride=8, dropout_p=None):
		super(ResNetMainClient, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		# inplanes and base width of the bottleneck
		if groups == 1:
			self.groups = groups
			inplanes_dict = {# 'imagenet': {1: 64, 2: 48, 4: 32, 8: 24},
								# 'imagenet': {1: 64, 2: 40, 4: 32, 8: 24},
								'imagenet': {1: 64, 2: 44, 4: 32, 8: 24},
								'cifar10': {1: 16, 2: 12, 4: 8, 8: 6, 16: 4, 32: 3},
								'cifar100': {1: 16, 2: 12, 4: 8, 8: 6, 16: 4},
								'svhn': {1: 16, 2: 12, 4: 8, 8: 6, 16: 4},
								'ham10000':{1: 64, 2: 44, 4: 32, 8: 24},
								'pill_base': {1: 64, 2: 44, 4: 32, 8: 24},
								'pill_large': {1: 64, 2: 44, 4: 32, 8: 24},
							}
			self.inplanes = inplanes_dict[dataset][split_factor]

			if 'wide_resnet' in arch:
				if arch in ['wide_resnetsl16_8']:
					self.inplanes = 16
				
				elif arch in ['wide_resnetsl50_2']:
					# wide_resnet50_2 and wide_resnet101_2 are for imagenet
					# assert split_factor in [1, 2, 4] and (dataset == 'imagenet' or dataset == 'ham10000' or dataset == 'pill_base' or dataset == 'pill_large')
					
					# The below is the same as max(widen_factor / (split_factor ** 0.5) + 0.4, 1.0)
					if arch == 'wide_resnetsl50_2' and split_factor == 2:
						self.inplanes = 64
						width_per_group = 64
						print('INFO:PyTorch: Dividing wide_resnet50_2, change base_width from {} '
								'to {}.'.format(64 * 2, 64))
					if arch == 'wide_resnet50_3' and split_factor == 2:
						self.inplanes = 64
						width_per_group = 64 * 2
						print('INFO:PyTorch: Dividing wide_resnet50_3, change base_width from {} '
								'to {}.'.format(64 * 3, 64 * 2))
				else:
					raise NotImplementedError

		elif groups in [8, 16, 32, 64]:
			# For resnext, just divide groups
			self.groups = groups
			if split_factor > 1:
				self.groups = int(groups / split_factor)
				print("INFO:PyTorch: Dividing {}, change groups from {} "
					"to {}.".format(arch, groups, self.groups))
			self.inplanes = 64
		
		else:
			raise NotImplementedError
		
		self.base_width = width_per_group
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None "
								"or a 3-element tuple, got {}".format(replace_stride_with_dilation))

		if (dataset == 'imagenet' or dataset == 'ham10000' or dataset == 'pill_base' or dataset == 'pill_large'):
			self.layer0 = nn.Sequential(
							nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
							norm_layer(self.inplanes),
							nn.ReLU(inplace=True),
							nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
							norm_layer(self.inplanes),
							nn.ReLU(inplace=True),
							# output channle = inplanes * 2
							nn.Conv2d(self.inplanes, self.inplanes * 2, kernel_size=3, stride=1, padding=1, bias=False),
							norm_layer(self.inplanes * 2),
							nn.ReLU(inplace=True),
							nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
						)
			inplanes_origin = self.inplanes
			# 64 -> 128
			self.inplanes = self.inplanes * 2
			strides = [1, 2, 2, 2]
			# n_channels = [64, 128, 256, 512]
		
		elif dataset in ['cifar10', 'cifar100', 'svhn']:
			# for training cifar, change the kernel_size=7 -> kernel_size=3 with stride=1
			self.layer0 = nn.Sequential(
							# nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
							nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
							norm_layer(self.inplanes),
							nn.ReLU(inplace=True),
						)
			inplanes_origin = self.inplanes
			
			if 'wide_resnet' in arch:
				widen_factor = float(arch.split('_')[-1])
				inplanes_origin = inplanes_origin * int(max(widen_factor / (split_factor ** 0.5) + 0.4, 1.0))

			# 32 -> 32 -> 16 -> 8
			strides = [1, 2, 2, 1]
			if output_stride == 2:
				print('INFO:PyTorch: Using output_stride {} on cifar10'.format(output_stride))
				strides = [1, 1, 2, 1]
		
		else:
			raise NotImplementedError

		# initialize the parameters
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				#nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, std=1e-3)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			if stride == 1:
				downsample = nn.Sequential(
					conv1x1(self.inplanes, planes * block.expansion, stride),
					norm_layer(planes * block.expansion),
				)
			else:
				downsample = nn.Sequential(
					# Ref:
					# Bag of Tricks for Image Classification with Convolutional Neural Networks, 2018
					# https://arxiv.org/abs/1812.01187
					# https://github.com/rwightman/pytorch-image-models/blob
					# /5966654052b24d99e4bfbcf1b59faae8a75db1fd/timm/models/resnet.py#L293
					#nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
					nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True, padding=0, count_include_pad=False),
					conv1x1(self.inplanes, planes * block.expansion, stride=1),
					norm_layer(planes * block.expansion),
				)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
							self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								norm_layer=norm_layer))

		return nn.Sequential(*layers)
	# Only forward until layer 0
	def _forward_impl(self, x):
		x = self.layer0(x)
		return x

	def forward(self, x):
		return self._forward_impl(x)

# Proxy clients model
class ResNetProxyClients(nn.Module):

	def __init__(self, arch, block, layers, num_classes=1000, zero_init_residual=True,
					groups=1, width_per_group=64, replace_stride_with_dilation=None,
					norm_layer=None, dataset='cifar10', split_factor=1, output_stride=8, dropout_p=None):
		super(ResNetProxyClients, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		# inplanes and base width of the bottleneck
		if groups == 1:
			self.groups = groups
			inplanes_dict = {# 'imagenet': {1: 64, 2: 48, 4: 32, 8: 24},
								# 'imagenet': {1: 64, 2: 40, 4: 32, 8: 24},
								'imagenet': {1: 64, 2: 44, 4: 32, 8: 24},
								'cifar10': {1: 16, 2: 12, 4: 8, 8: 6, 16: 4},
								'cifar100': {1: 16, 2: 12, 4: 8, 8: 6, 16: 4},
								'svhn': {1: 16, 2: 12, 4: 8, 8: 6, 16: 4},
								'ham10000':{1: 64, 2: 44, 4: 32, 8: 24},
								'pill_base': {1: 64, 2: 44, 4: 32, 8: 24},
								'pill_large': {1: 64, 2: 44, 4: 32, 8: 24},
							}
			self.inplanes = inplanes_dict[dataset][split_factor]
			
			if 'wide_resnet' in arch:
				if arch in ['wide_resnetsl16_8']:
					self.inplanes = 16
				
				elif arch in ['wide_resnetsl50_2']:
					# wide_resnet50_2 and wide_resnet101_2 are for imagenet
					# assert split_factor in [1, 2, 4] and (dataset == 'imagenet' or dataset=='ham10000' or dataset == 'pill_base' or dataset == 'pill_large')
					
					# The below is the same as max(widen_factor / (split_factor ** 0.5) + 0.4, 1.0)
					if arch == 'wide_resnetsl50_2' and split_factor == 2:
						self.inplanes = 64
						width_per_group = 64
						print('INFO:PyTorch: Dividing wide_resnet50_2, change base_width from {} '
								'to {}.'.format(64 * 2, 64))
					if arch == 'wide_resnet50_3' and split_factor == 2:
						self.inplanes = 64
						width_per_group = 64 * 2
						print('INFO:PyTorch: Dividing wide_resnet50_3, change base_width from {} '
								'to {}.'.format(64 * 3, 64 * 2))
				else:
					raise NotImplementedError

		elif groups in [8, 16, 32, 64]:
			# For resnext, just divide groups
			self.groups = groups
			if split_factor > 1:
				self.groups = int(groups / split_factor)
				print("INFO:PyTorch: Dividing {}, change groups from {} "
					"to {}.".format(arch, groups, self.groups))
			self.inplanes = 64
		
		else:
			raise NotImplementedError
		
		self.base_width = width_per_group
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None "
								"or a 3-element tuple, got {}".format(replace_stride_with_dilation))

		if (dataset == 'imagenet' or dataset == 'ham10000' or dataset == 'pill_base' or dataset == 'pill_large'):
			inplanes_origin = self.inplanes
			# 64 -> 128
			self.inplanes = self.inplanes * 2
			strides = [1, 2, 2, 2]
			# n_channels = [64, 128, 256, 512]
		
		elif dataset in ['cifar10', 'cifar100', 'svhn']:
			# for training cifar, change the kernel_size=7 -> kernel_size=3 with stride=1
			inplanes_origin = self.inplanes
			
			if 'wide_resnet' in arch:
				widen_factor = float(arch.split('_')[-1])
				inplanes_origin = inplanes_origin * int(max(widen_factor / (split_factor ** 0.5) + 0.4, 1.0))

			# 32 -> 32 -> 16 -> 8
			strides = [1, 2, 2, 1]
			if output_stride == 2:
				print('INFO:PyTorch: Using output_stride {} on cifar10'.format(output_stride))
				strides = [1, 1, 2, 1]
		
		else:
			raise NotImplementedError

		# For CIFAR, layer1 - layer3, channels - [16, 32, 64]
		# For CIFAR with ResNeXt, layer1 - layer3, channels - [64, 128, 256]
		# For ImageNet, layer1 - layer4, channels - [64, 128, 256, 512]
		self.layer1 = self._make_layer(block, inplanes_origin, layers[0], stride=strides[0])
		self.layer2 = self._make_layer(block, inplanes_origin * 2, layers[1], stride=strides[1],
										dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, inplanes_origin * 4, layers[2], stride=strides[2],
										dilate=replace_stride_with_dilation[1])
		inplanes_now = inplanes_origin * 4

		# If dataset is cifar, do not use layer4 because the size of the feature map is too small.
		# The original paper of resnet set total stride=8 with less channels.
		self.layer4 = None
		if ('imagenet' in dataset or 'ham10000' in dataset or 'pill_base' in dataset or 'pill_large' in dataset):
			print('INFO:PyTorch: Using layer4 for ImageNet Training')
			self.layer4 = self._make_layer(block, inplanes_origin * 8, layers[3], stride=strides[3],
													dilate=replace_stride_with_dilation[2])
			inplanes_now = inplanes_origin * 8
			
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		self.dropout = None
		if 'cifar' in dataset:
			if arch in ['wide_resnetsl16_8']:
				if dropout_p is not None:
					dropout_p = dropout_p / split_factor
					# You can also use the below code.
					# dropout_p = dropout_p / (split_factor ** 0.5)
					print('INFO:PyTorch: Using dropout with ratio {}'.format(dropout_p))
					self.dropout = nn.Dropout(dropout_p)

		elif 'imagenet' in dataset:
			if dropout_p is not None:
				dropout_p = dropout_p / split_factor
				# You can also use the below code.
				# dropout_p = dropout_p / (split_factor ** 0.5)
				print('INFO:PyTorch: Using dropout with ratio {}'.format(dropout_p))
				self.dropout = nn.Dropout(dropout_p)

		self.fc = nn.Linear(inplanes_now * block.expansion, num_classes)

		# initialize the parameters
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				#nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, std=1e-3)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			if stride == 1:
				downsample = nn.Sequential(
					conv1x1(self.inplanes, planes * block.expansion, stride),
					norm_layer(planes * block.expansion),
				)
			else:
				downsample = nn.Sequential(
					# Ref:
					# Bag of Tricks for Image Classification with Convolutional Neural Networks, 2018
					# https://arxiv.org/abs/1812.01187
					# https://github.com/rwightman/pytorch-image-models/blob
					# /5966654052b24d99e4bfbcf1b59faae8a75db1fd/timm/models/resnet.py#L293
					#nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
					nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True, padding=0, count_include_pad=False),
					conv1x1(self.inplanes, planes * block.expansion, stride=1),
					norm_layer(planes * block.expansion),
				)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
							self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								norm_layer=norm_layer))

		return nn.Sequential(*layers)
	# forward from layer 1
	def _forward_impl(self, x):
		# See note [TorchScript super()]
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		if self.layer4 is not None:
			x = self.layer4(x)

		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		
		#x = x.view(x.size(0), -1)
		if self.dropout is not None:
			x = self.dropout(x)
		
		x = self.fc(x)

		return x

	def forward(self, x):
		return self._forward_impl(x)
# main client model
def _resnetsl_main_client_(arch, block, layers, pretrained, progress, **kwargs):
	model_main_client = ResNetMainClient(arch, block, layers, **kwargs)
	return model_main_client
# proxy clients model
def _resnetsl_proxy_client_(arch, block, layers, pretrained, progress, **kwargs):
	model_proxy_clients = ResNetProxyClients(arch, block, layers, **kwargs)
	return model_proxy_clients
# Resnet 110 split 
def resnet110sl(pretrained=False, progress=True, **kwargs):
	r"""ResNet-110 model from
	`"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	assert 'cifar' in kwargs['dataset']
	return _resnetsl_main_client_('resnet110_sl', Bottleneck, [12, 12, 12, 12], pretrained, progress,
					**kwargs),_resnetsl_proxy_client_('resnet110_sl', Bottleneck, [12, 12, 12, 12], pretrained, progress,
					**kwargs)

def wide_resnetsl50_2(pretrained=False, progress=True, **kwargs):
    """Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

	The model is the same as ResNet except for the bottleneck number of channels
	which is twice larger in every block. The number of channels in outer 1x1
	convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
	channels, and in Wide ResNet-50-2 has 2048-1024-2048.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2

    return _resnetsl_main_client_('wide_resnetsl50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress,
					**kwargs),_resnetsl_proxy_client_('wide_resnetsl50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress,
					**kwargs)

# Note: This is a model for CIFAR dataset
def wide_resnetsl16_8(pretrained=False, progress=True, **kwargs):
	r"""Wide ResNet-16-8 model from
	`"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	kwargs['width_per_group'] = 64
	return _resnetsl_main_client_('wide_resnetsl16_8', BasicBlock, [2, 2, 2, 2],
						pretrained, progress, **kwargs),_resnetsl_proxy_client_('wide_resnetsl16_8', BasicBlock, [2, 2, 2, 2],
						pretrained, progress, **kwargs)
