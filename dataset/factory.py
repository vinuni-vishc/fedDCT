# coding=utf-8
"""
Get the dataloader for CIFAR, ImageNet, SVHN.
"""

import os
import random
from PIL import Image
import pickle
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from . import randaugment
from .svhn import SVHN
from .folder import ImageFolder
from .cifar import CIFAR10, CIFAR100
from .ham10000 import SkinData
from .autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
from .pill_dataset_base import PillDataBase
from .pill_dataset_large import PillDataLarge
from .cifar10_non_iid import *
from .cifar100_non_iid import *
__all__ = ['get_data_loader']


def get_data_loader(data_dir,
						split_factor=1,
						batch_size=128,
						crop_size=32,
						dataset='cifar10',
						split="train",
						is_distributed=False,
						is_autoaugment=1,
						randaa=None,
						is_cutout=True,
						erase_p=0.5,
						num_workers=8,
						pin_memory=True,
						is_fed=False,
						num_clusters=20,
						cifar10_non_iid=False,
						cifar100_non_iid = False):
	"""get the dataset loader"""
	assert not (is_autoaugment and randaa is not None)

	kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
	assert split in ['train', 'val', 'test']

	if dataset == 'cifar10':
		"""cifar10 dataset"""
		if (cifar10_non_iid == 'quantity_skew'):
			non_iid = 'quantity_skew'
			if 'train' in split:
				print("INFO:PyTorch: Using quantity_skew CIFAR10 dataset, batch size {} and crop size is {}.".format(batch_size, crop_size))
				traindir = os.path.join(data_dir, 'train')
				train_transform = transforms.Compose([transforms.ToPILImage(),
														transforms.RandomCrop(32, padding=4),
														transforms.RandomHorizontalFlip(),
														CIFAR10Policy(),
														transforms.ToTensor(),
														transforms.Normalize((0.4914, 0.4822, 0.4465),
														(0.2023, 0.1994, 0.2010)),
														transforms.RandomErasing(p=erase_p,
														scale=(0.125, 0.2),
														ratio=(0.99, 1.0),
														value=0, inplace=False),
				])
				train_sampler = None
				print('INFO:PyTorch: creating quantity_skew CIFAR10 train dataloader...')
				
				#print("Hey" + non_iid)
				if is_fed:
					train_loader = get_data_loaders_train( nclients= num_clusters*split_factor,
					batch_size=batch_size,verbose=True, transforms_train=train_transform,non_iid = non_iid,split_factor=split_factor)
				else:
					assert is_fed
				return train_loader, train_sampler
			else:
				valdir = os.path.join(data_dir, 'val')
				val_transform = transforms.Compose([
					transforms.ToPILImage(),
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465),
					(0.2023, 0.1994, 0.2010)),
					])
				val_loader = get_data_loaders_test( nclients= num_clusters*split_factor,
				batch_size=batch_size,verbose=True, transforms_eval=val_transform,non_iid = non_iid,split_factor=1)
				return val_loader 
		# elif (cifar10_non_iid == 'label_skew'):
		# 	non_iid = 'label_skew'
		# 	if 'train' in split:
		# 		print("INFO:PyTorch: Using label_skew CIFAR10 dataset, batch size {} and crop size is {}.".format(batch_size, crop_size))
		# 		traindir = os.path.join(data_dir, 'train')
		# 		train_transform = transforms.Compose([transforms.ToPILImage(),
		# 												transforms.RandomCrop(32, padding=4),
		# 												transforms.RandomHorizontalFlip(),
		# 												CIFAR10Policy(),
		# 												transforms.ToTensor(),
		# 												transforms.Normalize((0.4914, 0.4822, 0.4465),
		# 												(0.2023, 0.1994, 0.2010)),
		# 												transforms.RandomErasing(p=erase_p,
		# 												scale=(0.125, 0.2),
		# 												ratio=(0.99, 1.0),
		# 												value=0, inplace=False),
		# 		])
		# 		train_sampler = None
		# 		print('INFO:PyTorch: creating label_skew CIFAR10 train dataloader...')
		# 		non_iid = 'label_skew'
		# 		if is_fed:
		# 			train_loader = get_data_loaders_train(classes_pc=2, nclients= num_clusters*split_factor,
		# 			batch_size=batch_size,verbose=True, transforms_train=train_transform,non_iid = non_iid,split_factor=split_factor)
		# 		else:
		# 			assert is_fed
		# 		return train_loader, train_sampler
		# 	else:
		# 		valdir = os.path.join(data_dir, 'val')
		# 		val_transform = transforms.Compose([
		# 			transforms.ToPILImage(),
		# 			transforms.ToTensor(),
		# 			transforms.Normalize((0.4914, 0.4822, 0.4465),
		# 			(0.2023, 0.1994, 0.2010)),
		# 			])
		# 		val_loader = get_data_loaders_test( nclients= num_clusters*split_factor,
		# 		batch_size=batch_size,verbose=True, transforms_eval=val_transform,non_iid = non_iid,split_factor=1)
		# 		return val_loader 
		else:
			if 'train' in split:
				print("INFO:PyTorch: Using CIFAR10 dataset, batch size {} and crop size is {}.".format(batch_size, crop_size))
				traindir = os.path.join(data_dir, 'train')

				train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
														transforms.RandomHorizontalFlip(),
														CIFAR10Policy(),
														transforms.ToTensor(),
														transforms.Normalize((0.4914, 0.4822, 0.4465),
																			(0.2023, 0.1994, 0.2010)),
														transforms.RandomErasing(p=erase_p,
																				scale=(0.125, 0.2),
																				ratio=(0.99, 1.0),
																				value=0, inplace=False),
													])
				
				train_dataset = CIFAR10(traindir, train=True,
											transform=train_transform,
											target_transform=None,
											download=True,
											split_factor=split_factor)
				train_sampler = None
				

				if is_distributed:
					train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

				print('INFO:PyTorch: creating CIFAR10 train dataloader...')
				if is_fed:
					images_per_client = int(train_dataset.data.shape[0] / num_clusters/split_factor) 
					# print("data split is" +data_split)
					print("Images per client is "+ str(images_per_client))
					#traindata_split = torch.utils.data.random_split(train_dataset, [images_per_client for _ in range(num_clusters*split_factor)])
					data_split = [images_per_client for _ in range(num_clusters*split_factor-1)]
					
					data_split.append(len(train_dataset)-images_per_client*(num_clusters*split_factor-1))
					
					traindata_split = torch.utils.data.random_split(train_dataset,data_split,generator=torch.Generator().manual_seed(68))
					train_loader = [torch.utils.data.DataLoader(x,
																batch_size=batch_size,
																shuffle=(train_sampler is None),
																drop_last=True,
																sampler=train_sampler,
																**kwargs) for x in traindata_split]
				else:
					train_loader = torch.utils.data.DataLoader(train_dataset,
																batch_size=batch_size,
																shuffle=(train_sampler is None),
																drop_last=True,
																sampler=train_sampler,
																**kwargs)
				return train_loader, train_sampler
			else:
				valdir = os.path.join(data_dir, 'val')
				val_transform = transforms.Compose([transforms.ToTensor(),
													transforms.Normalize((0.4914, 0.4822, 0.4465),
																			(0.2023, 0.1994, 0.2010)),
													])
				val_dataset = CIFAR10(valdir, train=False,
										transform=val_transform,
										target_transform=None,
										download=True,
										split_factor=1)
				print('INFO:PyTorch: creating CIFAR10 validation dataloader...')
				val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
				return val_loader
	
	elif dataset == 'cifar100':
		"""cifar100 dataset"""
		if (cifar100_non_iid == 'quantity_skew'):
			non_iid = 'quantity_skew'
			if 'train' in split:
				print("INFO:PyTorch: Using quantity_skew CIFAR100 dataset, batch size {} and crop size is {}.".format(batch_size, crop_size))
				traindir = os.path.join(data_dir, 'train')
				train_transform = transforms.Compose([transforms.ToPILImage(),
														transforms.RandomCrop(32, padding=4),
														transforms.RandomHorizontalFlip(),
														CIFAR10Policy(),
														transforms.ToTensor(),
														transforms.Normalize((0.4914, 0.4822, 0.4465),
														(0.2023, 0.1994, 0.2010)),
														transforms.RandomErasing(p=erase_p,
														scale=(0.125, 0.2),
														ratio=(0.99, 1.0),
														value=0, inplace=False),
				])
				train_sampler = None
				print('INFO:PyTorch: creating quantity_skew CIFAR10 train dataloader...')
				
				#print("Hey" + non_iid)
				if is_fed:
					train_loader = get_data_loaders_train_cf100( nclients= num_clusters*split_factor,
					batch_size=batch_size,verbose=True, transforms_train=train_transform,non_iid = non_iid,split_factor=split_factor)
				else:
					assert is_fed
				return train_loader, train_sampler
			else:
				valdir = os.path.join(data_dir, 'val')
				val_transform = transforms.Compose([
					transforms.ToPILImage(),
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465),
					(0.2023, 0.1994, 0.2010)),
					])
				val_loader = get_data_loaders_test_cf100( nclients= num_clusters*split_factor,
				batch_size=batch_size,verbose=True, transforms_eval=val_transform,non_iid = non_iid,split_factor=1)
				return val_loader 
		else:
			if 'train' in split:
				print("INFO:PyTorch: Using CIFAR100 dataset, batch size {}"
						" and crop size is {}.".format(batch_size, crop_size))
				traindir = os.path.join(data_dir, 'train')
		
				train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
														transforms.RandomHorizontalFlip(),
														])
				if is_autoaugment:
					# the policy is the same as CIFAR10
					train_transform.transforms.append(CIFAR10Policy())
				train_transform.transforms.append(transforms.ToTensor())
				train_transform.transforms.append(transforms.Normalize((0.5071, 0.4865, 0.4409),
																		std=(0.2673, 0.2564, 0.2762)))
				
				if is_cutout:
					# use random erasing to mimic cutout
					train_transform.transforms.append(transforms.RandomErasing(p=erase_p,
																				scale=(0.0625, 0.1),
																				ratio=(0.99, 1.0),
																				value=0, inplace=False))
				train_dataset = CIFAR100(traindir, train=True,
											transform=train_transform,
											target_transform=None,
											download=True,
											split_factor=split_factor)
				train_sampler = None
				if is_distributed:
					train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

				print('INFO:PyTorch: creating CIFAR100 train dataloader...')
				if is_fed:
					images_per_client = int(train_dataset.data.shape[0] / num_clusters/split_factor) 
					print("Images per client is "+ str(images_per_client))
					data_split = [images_per_client for _ in range(num_clusters*split_factor-1)]
						
					data_split.append(len(train_dataset)-images_per_client*(num_clusters*split_factor-1))
						
					traindata_split = torch.utils.data.random_split(train_dataset,data_split,generator=torch.Generator().manual_seed(68))
					train_loader = [torch.utils.data.DataLoader(x,
																batch_size=batch_size,
																shuffle=(train_sampler is None),
																drop_last=True,
																sampler=train_sampler,
																**kwargs) for x in traindata_split]
				else:
					train_loader = torch.utils.data.DataLoader(train_dataset,
															batch_size=batch_size,
															shuffle=(train_sampler is None),
															drop_last=True,
															sampler=train_sampler,
															**kwargs)
				return train_loader, train_sampler
			else:
				valdir = os.path.join(data_dir, 'val')
				val_transform = transforms.Compose([transforms.ToTensor(),
													transforms.Normalize((0.5071, 0.4865, 0.4409),
																			std=(0.2673, 0.2564, 0.2762)),
													])
				val_dataset = CIFAR100(valdir, train=False,
										transform=val_transform,
										target_transform=None,
										download=True,
										split_factor=1)
				print('INFO:PyTorch: creating CIFAR100 validation dataloader...')
				val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
				return val_loader
	
	elif dataset == 'imagenet':
			
		if 'train' in split:
			print("INFO:PyTorch: Using ImageNet dataset, batch size {} and crop size is {}.".format(batch_size, crop_size))
			traindir = os.path.join(data_dir, 'train')
			mean = [0.485, 0.456, 0.406]
			train_transform = transforms.Compose([transforms.RandomResizedCrop(crop_size, scale=(0.1, 1.0),
																				interpolation=Image.BICUBIC),
														transforms.RandomHorizontalFlip()
													])
			# AutoAugment
			if is_autoaugment:
				print('INFO:PyTorch: creating ImageNet AutoAugment policies')
				train_transform.transforms.append(ImageNetPolicy())
			# RandAugment
			if randaa is not None and randaa.startswith('rand'):
				print('INFO:PyTorch: creating ImageNet RandAugment policies')
				aa_params = dict(
								translate_const=int(crop_size * 0.45),
								img_mean=tuple([min(255, round(255 * x)) for x in mean]),
							)
				train_transform.transforms.append(
										randaugment.rand_augment_transform(randaa, aa_params))
			# To tensor and normalize
			train_transform.transforms.append(transforms.ToTensor())
			train_transform.transforms.append(transforms.Normalize([0.485, 0.456, 0.406],
																	std=[0.229, 0.224, 0.225]))

			if is_cutout:
				print('INFO:PyTorch: creating RandomErasing policies...')
				train_transform.transforms.append(transforms.RandomErasing(p=erase_p, scale=(0.05, 0.12),
																			ratio=(0.5, 1.5), value=0))

			train_dataset = ImageFolder(traindir, transform=train_transform, split_factor=split_factor)
			
			train_sampler = None
			if is_distributed:
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

			print('INFO:PyTorch: creating ImageNet train dataloader...')
			train_loader = torch.utils.data.DataLoader(train_dataset,
														batch_size=batch_size,
														shuffle=(train_sampler is None),
														drop_last=True,
														sampler=train_sampler,
														**kwargs)
			return train_loader, train_sampler
		else:
			valdir = os.path.join(data_dir, 'val')
			val_transform = transforms.Compose([transforms.Resize(int(crop_size * 1.15),
																	interpolation=Image.BICUBIC),
												transforms.CenterCrop(crop_size),
												transforms.ToTensor(),
												transforms.Normalize([0.485, 0.456, 0.406],
																		std=[0.229, 0.224, 0.225]),
											])
			
			val_dataset = ImageFolder(valdir, transform=val_transform, split_factor=1)

			print('INFO:PyTorch: creating ImageNet validation dataloader...')
			val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
			return val_loader
	elif dataset == 'pill_large':
		mean = [0.4807, 0.5434, 0.5801]
		std = [0.2443, 0.2398, 0.2222]			
		if 'train' in split:
			print("INFO:PyTorch: Using pill base dataset, batch size {} and crop size is {}.".format(batch_size, crop_size))
			train_transform = transforms.Compose([transforms.RandomResizedCrop(crop_size, scale=(0.1, 1.0),
																				interpolation=Image.BILINEAR),
														transforms.RandomHorizontalFlip()
													])
			# RandAugment
			if randaa is not None and randaa.startswith('rand'):
				print('INFO:PyTorch: creating Pill Large RandAugment policies')
				aa_params = dict(
								translate_const=int(crop_size * 0.45),
								img_mean=tuple([min(255, round(255 * x)) for x in mean]),
							)
				train_transform.transforms.append(
										randaugment.rand_augment_transform(randaa, aa_params))
			# To tensor and normalize
			train_transform.transforms.append(transforms.ToTensor())
			train_transform.transforms.append(transforms.Normalize(mean,
																	std))
			if is_cutout:
				print('INFO:PyTorch: creating RandomErasing policies...')
				train_transform.transforms.append(transforms.RandomErasing(p=erase_p, scale=(0.05, 0.12),
																			ratio=(0.5, 1.5), value=0))
			train_dataset = PillDataLarge(True,transform=train_transform,split_factor=split_factor)
			train_sampler = None
			if is_distributed:
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

			print('INFO:PyTorch: creating Pill Large train dataloader...')
			if is_fed:
				images_per_client = int(len(train_dataset) / num_clusters/split_factor) 
				print("Images per client is "+ str(images_per_client))
				data_split = [images_per_client for _ in range(num_clusters*split_factor-1)]
				data_split.append(len(train_dataset)-images_per_client*(num_clusters*split_factor-1))
				#print(data_split)
				traindata_split = torch.utils.data.random_split(train_dataset,data_split,generator=torch.Generator().manual_seed(68))
				#traindata_split = torch.utils.data.random_split(train_dataset, [images_per_client for _ in range(num_clusters*split_factor)])
				train_loader = [torch.utils.data.DataLoader(x,
															batch_size=batch_size,
															shuffle=(train_sampler is None),
															drop_last=True,
															sampler=train_sampler,
															**kwargs) for x in traindata_split]
			else:
				train_loader = torch.utils.data.DataLoader(train_dataset,
														batch_size=batch_size,
														shuffle=(train_sampler is None),
														drop_last=True,
														sampler=train_sampler,
														**kwargs)
			return train_loader, train_sampler

		else:
			valdir = os.path.join(data_dir, 'val')
			val_transform = transforms.Compose([transforms.Resize(int(crop_size * 1.15),
																	interpolation=Image.BILINEAR),
												transforms.CenterCrop(crop_size),
												transforms.ToTensor(),
												transforms.Normalize(mean,
																		std),
											])
			
			val_dataset = PillDataLarge(False,transform=val_transform,split_factor=1)

			print('INFO:PyTorch: creating Pill Large validation dataloader...')
			val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
			return val_loader

	elif dataset == 'pill_base':
		mean = [0.4550, 0.5239, 0.5653]
		std = [0.2460, 0.2446, 0.2252]			
		if 'train' in split:
			print("INFO:PyTorch: Using pill Base dataset, batch size {} and crop size is {}.".format(batch_size, crop_size))
			train_transform = transforms.Compose([transforms.RandomResizedCrop(crop_size, scale=(0.1, 1.0),
																				interpolation=Image.BILINEAR),
														transforms.RandomHorizontalFlip()
													])
			# RandAugment
			if randaa is not None and randaa.startswith('rand'):
				print('INFO:PyTorch: creating Pill Base RandAugment policies')
				aa_params = dict(
								translate_const=int(crop_size * 0.45),
								img_mean=tuple([min(255, round(255 * x)) for x in mean]),
							)
				train_transform.transforms.append(
										randaugment.rand_augment_transform(randaa, aa_params))
			# To tensor and normalize
			train_transform.transforms.append(transforms.ToTensor())
			train_transform.transforms.append(transforms.Normalize(mean,
																	std))
			if is_cutout:
				print('INFO:PyTorch: creating RandomErasing policies...')
				train_transform.transforms.append(transforms.RandomErasing(p=erase_p, scale=(0.05, 0.12),
																			ratio=(0.5, 1.5), value=0))
			train_dataset = PillDataBase(True,transform=train_transform,split_factor=split_factor)
			
			train_sampler = None
			if is_distributed:
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

			print('INFO:PyTorch: creating Pill Base train dataloader...')
			if is_fed:
				images_per_client = int(len(train_dataset) / num_clusters/split_factor) 
				print("Images per client is "+ str(images_per_client))
				data_split = [images_per_client for _ in range(num_clusters*split_factor-1)]
				data_split.append(len(train_dataset)-images_per_client*(num_clusters*split_factor-1))
				traindata_split = torch.utils.data.random_split(train_dataset,data_split,generator=torch.Generator().manual_seed(68))
				train_loader = [torch.utils.data.DataLoader(x,
															batch_size=batch_size,
															shuffle=(train_sampler is None),
															drop_last=True,
															sampler=train_sampler,
															**kwargs) for x in traindata_split]
			else:
				train_loader = torch.utils.data.DataLoader(train_dataset,
														batch_size=batch_size,
														shuffle=(train_sampler is None),
														drop_last=True,
														sampler=train_sampler,
														**kwargs)
			return train_loader, train_sampler

		else:
			valdir = os.path.join(data_dir, 'val')
			val_transform = transforms.Compose([transforms.Resize(int(crop_size * 1.15),
																	interpolation=Image.BILINEAR),
												transforms.CenterCrop(crop_size),
												transforms.ToTensor(),
												transforms.Normalize(mean,
																		std),
											])
			
			val_dataset = PillDataBase(False,transform=val_transform,split_factor=1)

			print('INFO:PyTorch: creating Pill Base validation dataloader...')
			val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
			return val_loader
	elif dataset == 'ham10000':
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		if 'train' in split:
			print("INFO:PyTorch: Using ham10000 dataset, batch size {} and crop size is {}.".format(batch_size, crop_size))
			train_file = open(HOME+'/dataset/ham10000/ham10000_train.pickle','rb')
			train = pickle.load(train_file)
			train_file.close
			train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                        transforms.RandomVerticalFlip(),
						transforms.RandomHorizontalFlip(), #new
						transforms.RandomAdjustSharpness(random.uniform(0, 4.0)),
						transforms.RandomAutocontrast(),
                        transforms.Pad(3),
                        transforms.RandomRotation(10),
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = mean, std = std)
                        ])
			
			train_dataset = SkinData(train, transform = train_transforms,split_factor=split_factor)
			train_sampler = None
			if is_distributed:
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

			print('INFO:PyTorch: creating ImageNet train dataloader...')
			if is_fed:
				images_per_client = int(len(train_dataset)/ num_clusters/split_factor) 
				print("Images per client is "+ str(images_per_client))
				data_split = [images_per_client for _ in range(num_clusters*split_factor-1)]
				data_split.append(len(train_dataset)-images_per_client*(num_clusters*split_factor-1))
				#print(data_split)
				traindata_split = torch.utils.data.random_split(train_dataset,data_split,generator=torch.Generator().manual_seed(68))
				train_loader = [torch.utils.data.DataLoader(x,
															batch_size=batch_size,
															shuffle=(train_sampler is None),
															drop_last=True,
															sampler=train_sampler,
															**kwargs) for x in traindata_split]
			else:
				train_loader = torch.utils.data.DataLoader(train_dataset,
															batch_size=batch_size,
															shuffle=(train_sampler is None),
															drop_last=True,
															sampler=train_sampler,
															**kwargs)
			return train_loader, train_sampler

		else:
			test_file = open(HOME+"/dataset/ham10000/ham10000_test.pickle","rb")
			test = pickle.load(test_file)
			test_file.close
			val_transforms = transforms.Compose([
                        transforms.Pad(3),
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = mean, std = std)
                        ]) 
			val_dataset = SkinData(test, transform = val_transforms,split_factor=split_factor)   
			val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
			return val_loader
			
	elif dataset == 'svhn':
		"""SVHN dataset"""
		
		if 'train' in split:
			print("INFO:PyTorch: Using SVHN dataset, batch size {}"
					"and crop size is {}.".format(batch_size, crop_size))
			train_transform = transforms.Compose([transforms.RandomCrop(crop_size, padding=4),
													transforms.RandomHorizontalFlip(),
													SVHNPolicy(),
													transforms.ToTensor(),
													transforms.Normalize((0.4309, 0.4302, 0.4463),
																			std=(0.1965, 0.1983, 0.1994)),
													transforms.RandomErasing(p=erase_p,
																				scale=(0.125, 0.25),
																				ratio=(0.9, 1.1),
																				value=0, inplace=False),
													# Cutout(1, length=20),
												])
			train_dataset_01 = SVHN(data_dir, split='train',
										transform=train_transform,
										target_transform=None,
										download=True,
										split_factor=split_factor)
			
			train_dataset_02 = SVHN(data_dir, split='extra',
										transform=train_transform,
										target_transform=None,
										download=True,
										split_factor=split_factor)

			train_dataset = torch.utils.data.ConcatDataset([train_dataset_01, train_dataset_02])
			
			train_sampler = None
			if is_distributed:
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

			print('INFO:PyTorch: creating SVHN train dataloader...')
			train_loader = torch.utils.data.DataLoader(train_dataset,
														batch_size=batch_size,
														shuffle=(train_sampler is None),
														drop_last=True,
														sampler=train_sampler,
														**kwargs)
			return train_loader, train_sampler
		else:
			val_transform = transforms.Compose([transforms.ToTensor(),
												transforms.Normalize((0.4309, 0.4302, 0.4463),
																		std=(0.1965, 0.1983, 0.1994)),
												])
			val_dataset = SVHN(data_dir, split='test',
									transform=val_transform,
									target_transform=None,
									download=True,
									split_factor=1)
			print('INFO:PyTorch: creating SVHN validation dataloader...')
			val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
			return val_loader
	else:
		"""raise error"""
		raise NotImplementedError("The DataLoader for {} is not implemented.".format(dataset))


def calculate_svhn_mean_std():
	# data dir
	import numpy as np
	data_dir = os.path.join("/home/zhaoshuai/dataset/svhn")
	print(data_dir)
	train_dataset_01 = SVHN(data_dir,
								split='train',
								transform=None,
								target_transform=None,
								download=True)
	train_dataset_02 = SVHN(data_dir, split='extra',
										transform=None,
										target_transform=None,
										download=True)
	# train_dataset = torch.utils.data.ConcatDataset([train_dataset_01, train_dataset_02])
	train_data = np.concatenate([train_dataset_01.data, train_dataset_02.data], axis=0)

	image_mean = np.array([0.0, 0.0, 0.0])
	cov_sum = np.array([0.0, 0.0, 0.0])
	pixel_nums = 0.0
	
	# mean
	num_images = len(train_data)
	print("The total number of images is {}".format(num_images))
	for i in range(num_images):
		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(np.transpose(train_data[i], (1, 2, 0)))

		image = np.array(img).astype(np.float32)
		pixel_nums += image.shape[0] * image.shape[1]
		image_mean += np.sum(image, axis=(0, 1))
	
	image_mean = image_mean / pixel_nums
	print("The mean of SVHN dataset is: {}".format(image_mean))
	
	# std
	for i in range(num_images):
		img = Image.fromarray(np.transpose(train_data[i], (1, 2, 0)))

		image = np.array(img).astype(np.float32)
		cov_sum += np.sum(np.square(image - image_mean), axis=(0, 1))
	
	image_cov = np.sqrt(cov_sum / (pixel_nums - 1))
	print("The std of SVHN dataset is: {}".format(image_cov))



if __name__ == '__main__':
	"""check the imagenet dataset"""
	"""
	import warnings
	warnings.filterwarnings("error", category=UserWarning)
	print('INFO:PyTorch: creating ImageNet train dataset...')
	traindir = '/home/zhaoshuai/dataset/imagenet/train'
	train_dataset = datasets.ImageFolder(traindir, transform=None)
	i = 1002033
	while i < len(train_dataset.imgs):
		(img_path, idx) = train_dataset.imgs[i]
		try:
			result = Image.open(img_path).convert("RGB")
			result.verify()
			print("{}-th: Image {} is normal".format(i, img_path))
			i += 1
		except:
			print(img_path)
			raise IOError
	"""
	# corrupt
	# /home/zhaoshuai/dataset/imagenet/train/n04152593/n04152593_17460.JPEG
	
	"""calculate the mean and std of SVHN dataset"""
	calculate_svhn_mean_std()
	# mean [109.8819638,  109.71191519, 113.8176098]
	# std [50.11478077, 50.57169709, 50.85229309]
