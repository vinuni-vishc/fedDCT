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
from .folder import ImageFolder
from .cifar import CIFAR10, CIFAR100
from .ham10000 import SkinData
from .autoaugment import CIFAR10Policy
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
				# traindir = os.path.join(data_dir, 'train')
				traindir = data_dir
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
					train_loader = get_data_loaders_train( traindir, nclients= num_clusters*split_factor,
					batch_size=batch_size,verbose=True, transforms_train=train_transform,non_iid = non_iid,split_factor=split_factor)
					print(train_loader)
				else:
					assert is_fed
				return train_loader, train_sampler
			else:
				# valdir = os.path.join(data_dir, 'val')
				valdir = data_dir
				val_transform = transforms.Compose([
					transforms.ToPILImage(),
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465),
					(0.2023, 0.1994, 0.2010)),
					])
				val_loader = get_data_loaders_test(valdir, nclients= num_clusters*split_factor,
				batch_size=batch_size,verbose=True, transforms_eval=val_transform,non_iid = non_iid,split_factor=1)
				return val_loader 
		else:
			if 'train' in split:
				print("INFO:PyTorch: Using CIFAR10 dataset, batch size {} and crop size is {}.".format(batch_size, crop_size))
				# traindir = os.path.join(data_dir, 'train')
				traindir = data_dir
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
				# valdir = os.path.join(data_dir, 'val')
				valdir = data_dir
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
				# traindir = os.path.join(data_dir, 'train')
				# traindir = data_dir
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
					train_loader = get_data_loaders_train_cf100(data_dir, nclients= num_clusters*split_factor,
					batch_size=batch_size,verbose=True, transforms_train=train_transform,non_iid = non_iid,split_factor=split_factor)
				else:
					assert is_fed
				return train_loader, train_sampler
			else:
				# valdir = os.path.join(data_dir, 'val')
				# valdir = data_dir
				val_transform = transforms.Compose([
					transforms.ToPILImage(),
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465),
					(0.2023, 0.1994, 0.2010)),
					])
				val_loader = get_data_loaders_test_cf100(data_dir, nclients= num_clusters*split_factor,
				batch_size=batch_size,verbose=True, transforms_eval=val_transform,non_iid = non_iid,split_factor=1)
				return val_loader 
		else:
			if 'train' in split:
				print("INFO:PyTorch: Using CIFAR100 dataset, batch size {}"
						" and crop size is {}.".format(batch_size, crop_size))
				# traindir = os.path.join(data_dir, 'train')
				traindir = data_dir
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
				# valdir = os.path.join(data_dir, 'val')
				valdir = data_dir
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
			print("INFO:PyTorch: Using pill Large dataset, batch size {} and crop size is {}.".format(batch_size, crop_size))
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
			train_dataset = PillDataBase(data_dir,True,transform=train_transform,split_factor=split_factor)

			train_sampler = None
			if is_distributed:
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

			print('INFO:PyTorch: creating Pill Large train dataloader...')
			if is_fed:
				print("len of ds "+str(len(train_dataset)))
				# images_per_client = int(len(train_dataset) / num_clusters/split_factor) 
				# print("Images per client is "+ str(images_per_client))
				# data_split = [images_per_client for _ in range(num_clusters*split_factor-1)]
				# data_split.append(len(train_dataset)-images_per_client*(num_clusters*split_factor-1))
				data_split = [244,571,326,489,163,652,326,326,571,408,408,816,204,204,652,163,571,244,489,334]
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

			val_dataset = PillDataBase(data_dir,False,transform=val_transform,split_factor=1)

			print('INFO:PyTorch: creating Pill Large validation dataloader...')
			val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
			return val_loader
	elif dataset == 'ham10000':
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		train = [[] for _ in range(20)]
		train_dataset=[]
		train_loader = []
		if 'train' in split:
			print("INFO:PyTorch: Using ham10000 dataset, batch size {} and crop size is {}.".format(batch_size, crop_size))
			# train_file = open(HOME+'/dataset/ham10000/ham10000_train.pickle','rb')
			# train = pickle.load(train_file)
			# train_file.close
			for client_num in range(1, 21):
				print("load dataset for client "+ str(client_num))
				train_file= open( HOME+'/dataset/ham10000/' + f'client{client_num}.pickle','rb')
				
				train[client_num - 1] = pickle.load(train_file)
				print(train[client_num - 1])
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
			for client_num in range(1, 21):
				train_dataset.append(SkinData(train[client_num-1], transform = train_transforms,split_factor=split_factor))
			train_sampler = None
			if is_distributed:
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
			# print(len(train_dataset))
			print('INFO:PyTorch: creating ImageNet train dataloader...')
			if is_fed:
				for client_num in range(1, 21):
					train_loader.append(torch.utils.data.DataLoader(train_dataset[client_num-1],
															batch_size=batch_size,
															shuffle=(train_sampler is None),
															drop_last=True,
															sampler=train_sampler,
															**kwargs))

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
			
	else:
		"""raise error"""
		raise NotImplementedError("The DataLoader for {} is not implemented.".format(dataset))


