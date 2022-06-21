import os
from sklearn import ensemble
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mixup import mixup_criterion,mixup_data
from . import resnet,resnet_sl
__all__ = ['SplitNet']

def _get_net(arch='wide_resnet28_10'):
    networks_obj_dict = {
		# For ImageNet
        # For CIFAR/SVHN
        'wide_resnet28_10': resnet.wide_resnet28_10,
        'wide_resnet16_8': resnet.wide_resnet16_8,
        'resnet110': resnet.resnet110,    
        'wide_resnet50_2': resnet.wide_resnet50_2,
        'resnet110sl': resnet_sl.resnet110sl,   
        'wide_resnetsl50_2': resnet_sl.wide_resnetsl50_2,
        'wide_resnetsl16_8': resnet_sl.wide_resnetsl16_8,  
    }
    assert arch in networks_obj_dict.keys()
    return networks_obj_dict[arch]

class SplitNet(nn.Module):
    def __init__(self,
                    args,
                    norm_layer = None,
                    criterion=None,
                    progress=True):
        super(SplitNet, self).__init__()
        self.split_factor = args.split_factor
        self.arch = args.arch
        self.loop_factor = args.loop_factor
        self.is_train_sep = args.is_train_sep
        self.epochs = args.epochs
        # create models
        
        if self.arch in ['resnet50', 'resnet101', 'resnet152', 'resnet200',
							'resnext50_32x4d', 'resnext101_32x4d',
							'resnext101_64x4d',
							'resnext29_8x64d', 'resnext29_16x64d',
							'resnet110', 'resnet164',
							'wide_resnet16_8', 'wide_resnet16_12',
							'wide_resnet28_10', 'wide_resnet40_10', 'wide_resnet52_8',
							'wide_resnet50_2', 'wide_resnet50_3', 'wide_resnet101_2']:
			
            model_kwargs = {'num_classes': args.num_classes,
							'norm_layer': norm_layer,
							'dataset': args.dataset,
							'split_factor': self.split_factor,
							'output_stride': args.output_stride,
							}
            models = []
            for i in range(self.loop_factor):
                models.append(_get_net(self.arch)(pretrained=args.pretrained, **model_kwargs))
            self.models = nn.ModuleList(models)

        else:
                raise NotImplementedError
        # Holds submodules in a list. Docs: https://pytorch.org/docs/stable/nn.html#modulelist
        
        self.criterion = criterion
        if args.is_identical_init:
            print("INFO:PyTorch: Using identical initialization.")
            self._identical_init()
        # data transform - use different transformers for different networks
        self.is_diff_data_train = args.is_diff_data_train
        self.is_mixup = args.is_mixup
        self.mix_alpha = args.mix_alpha   

        # add loss of the ensembled output
        self.is_ensembled_loss = False if self.split_factor <= 1 else args.is_ensembled_loss
        self.ensembled_loss_weight = args.ensembled_loss_weight
        self.is_ensembled_after_softmax = False if self.split_factor <= 1 else args.is_ensembled_after_softmax
        self.is_max_ensemble = False if self.split_factor <= 1 else args.is_max_ensemble

        # using cot_loss (co-training loss)
        self.is_cot_loss = False if self.split_factor <= 1 else args.is_cot_loss
        self.cot_weight = args.cot_weight
        self.is_cot_weight_warm_up = args.is_cot_weight_warm_up
        self.cot_weight_warm_up_epochs = args.cot_weight_warm_up_epochs
        # self.kl_temperature = args.kl_temperature
        self.cot_loss_choose = args.cot_loss_choose
        print("INFO:PyTorch: The co-training loss is {}.".format(self.cot_loss_choose))
        self.num_classes = args.num_classes

    def forward(self, x, target=None, mode='train', epoch=0, streams=None):
        """
		Output:
			ensemble_output: 	a tensor of shape [batch_size, num_classes]
			outputs: 			a tensor of shape [split_factor, batch_size, num_classes]
			ce_losses: 			a tensor of shape [split_factor, ]
			cot_loss: 			a tensor of shape [split_factor, ]
		"""
        if self.arch in ['resnet50', 'resnet101', 'resnet152', 'resnet200',
							'resnext50_32x4d', 'resnext101_32x4d',
							'resnext101_64x4d',
							'resnext29_8x64d', 'resnext29_16x64d',
							'resnet110', 'resnet164',
							'wide_resnet16_8', 'wide_resnet16_12',
							'wide_resnet28_10', 'wide_resnet40_10', 'wide_resnet52_8',
							'wide_resnet50_2', 'wide_resnet50_3', 'wide_resnet101_2']:
            outputs = []
            ce_losses = []
            
            if 'train' in mode:
                # Only do mixup for one time, otherwise it will violate the co-training loss and ensembled loss
                if self.is_mixup:
                    x, y_a, y_b, lam = mixup_data(x, target, alpha=self.mix_alpha)
                if self.is_diff_data_train:
                    # split the tensor of size [N, split_factor * C, H, W] into several [N, C, H, W] tensors
                    all_x = torch.chunk(x, chunks=self.loop_factor, dim=1)
                # How to run loop in parallel?
                # https://discuss.pytorch.org/t/using-streams-doesnt-seem-to-improve-performance/28575
                # https://discuss.pytorch.org/t/parallelize-simple-for-loop-for-single-gpu/33701/5
                # https://stackoverflow.com/questions/52498690/how-to-use-cuda-stream-in-pytorch
                # https://pytorch.org/docs/master/notes/multiprocessing.html
                if self.loop_factor == 1 or streams is None:
                    for i in range(self.loop_factor):
                        x_tmp = all_x[i] if self.is_diff_data_train else x
                        output = self.models[i](x_tmp)
                        # calculate the loss
                        loss_now = mixup_criterion(self.criterion, output, y_a, y_b, lam) \
                                            if self.is_mixup else self.criterion(output, target)
                        ce_losses.append(loss_now)
                        outputs.append(output)
                else:
                    raise NotImplementedError

            elif mode in ['val','test']:
                for i in range (self.loop_factor):
                    output = self.models[i](x)
                    if self.criterion is not None:
                        loss_now = self.criterion(output,target)
                    else:
                        loss_now = torch.zeros(1)
                    outputs.append(output)
                    ce_losses.append(loss_now)
            else:
                # for model summary
                for i in range(self.loop_factor):
                    output = self.models[i](x)
                return torch.ones(1)
            
            #calculate the ensemble output
            emsemble_output = self._get_ensemble_output(outputs)
            ce_loss = torch.sum(torch.stack(ce_losses,dim =0))

            if 'val' in mode or 'test' in mode:
                return emsemble_output, torch.stack(outputs, dim=0), ce_loss
            # js divergence between all outputs
            if self.is_cot_loss:
                cot_loss = self._co_training_loss(outputs, self.cot_loss_choose, epoch=epoch)
            else:
                cot_loss = torch.zeros_like(ce_loss)
            return emsemble_output, torch.stack(outputs, dim=0), ce_loss, cot_loss

        else:
                raise NotImplementedError
    def _get_ensemble_output(self, outputs):
        """
		calculate the ensemble output.
		Currently, it only supports simple average or max.
		Args:
			outputs: a list of tensors, len(outputs) = split_factor.
		"""
        if self.is_ensembled_after_softmax:
            if self.is_max_ensemble:
                ensemble_output, _ = torch.max(F.softmax(torch.stack(outputs, dim=0), dim=-1), dim=0)
            else:
                ensemble_output = torch.mean(F.softmax(torch.stack(outputs, dim=0), dim=-1), dim=0)
            
        else: 
            if self.is_max_ensemble:
                ensemble_output, _ = torch.max(torch.stack(outputs,dim=0),dim =0 )
            else:
                ensemble_output = torch.mean(torch.stack(outputs,dim = 0),dim = 0)
        return ensemble_output
    
    def _co_training_loss(self,outputs,loss_choose,epoch=0):
        """calculate the co-training loss between outputs of different small networks
		"""
        weight_now = self.cot_weight
        if self.is_cot_weight_warm_up and epoch < self.cot_weight_warm_up_epochs:
            weight_now = max(self.cot_weight * epoch / self.cot_weight_warm_up_epochs, 0.005)
        if loss_choose == 'js_divergence':
            # the Jensen-Shannon divergence between p(x1), p(x2), p(x3)...
			# https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
            outputs_all = torch.stack(outputs, dim=0)
            p_all = F.softmax(outputs_all, dim=-1)
            p_mean = torch.mean(p_all, dim=0)
            H_mean = (- p_mean * torch.log(p_mean)).sum(-1).mean()
            H_sep = (- p_all * F.log_softmax(outputs_all, dim=-1)).sum(-1).mean()
            cot_loss = weight_now * (H_mean - H_sep)

        elif loss_choose == 'kl_seperate':
            outputs_all = torch.stack(outputs, dim=0)
            # repeat [1,2,3] like [1,1,2,2,3,3] and [2,3,1,3,1,2]
            outputs_r1 = torch.repeat_interleave(outputs_all, self.split_factor - 1, dim=0)
            index_list = [j for i in range(self.split_factor) for j in range(self.split_factor) if j!=i]
            outputs_r2 = torch.index_select(outputs_all, dim=0, index=torch.tensor(index_list, dtype=torch.long).cuda())
			# calculate the KL divergence
            kl_loss = F.kl_div(F.log_softmax(outputs_r1, dim=-1),
												F.softmax(outputs_r2, dim=-1).detach(),
												reduction='none')
			# cot_loss = weight_now * (kl_loss.sum(-1).mean(-1).sum() / (self.split_factor - 1))
            cot_loss = weight_now * (kl_loss.sum(-1).mean(-1).sum() / (self.split_factor - 1))
		
        else:
            raise NotImplementedError

        return cot_loss

    def _identical_init(self):
        """make the initial weights of networks the same"""
        with torch.no_grad():
            for i in range(1, self.split_factor):
                for (name1, m1), (name2, m2) in zip(self.models[i].named_parameters(),
														self.models[0].named_parameters()):
                    if 'weight' in name1:
                        m1.data.copy_(m2.data)