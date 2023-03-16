from http import client
import os
import sys
import time
import random
import shutil
import argparse
import warnings
import setproctitle
import json
import torch
import torch.cuda.amp as amp
from torch import nn, distributed
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from config import *
from params import train_params
from utils import label_smoothing, norm, summary, metric, lr_scheduler, rmsprop_tf, prefetch
from model import splitnet,splitnetsl
from utils.thop import profile, clever_format
from dataset import factory
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from params.train_params import save_hp_to_json
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# global best accuracy
best_acc1 = 0

def Average(lst):
    return sum(lst) / len(lst)

def server_aggregate(global_model_main_client,global_model_proxy_clients,models_main_client, models_proxy_clients):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_main_client_dict = global_model_main_client.state_dict()
    global_model_proxy_clients_dict = global_model_proxy_clients.state_dict()

    for k in global_main_client_dict.keys():
        global_main_client_dict[k] = torch.stack([models_main_client[i].state_dict(
        )[k].float() for i in range(len(models_main_client))], 0).mean(0)
    global_model_main_client.load_state_dict(global_main_client_dict)

    for k in global_model_proxy_clients_dict.keys():
        global_model_proxy_clients_dict[k] = torch.stack([models_proxy_clients[i].state_dict(
        )[k].float() for i in range(len(models_proxy_clients))], 0).mean(0)
    global_model_proxy_clients.load_state_dict(global_model_proxy_clients_dict)
    
    for model in models_main_client:
        model.load_state_dict(global_model_main_client.state_dict())

    for model in models_proxy_clients:
        model.load_state_dict(global_model_proxy_clients.state_dict())

def client_update(args, current_round, model_main_client,model_proxy_clients, scheduler_main_client,scheduler_proxy_clients, optimizer_main_client,optimizer_proxy_clients, train_loader, epoch=5, streams=None):
    model_main_client.train()
    model_proxy_clients.train()
    for e in range(epoch):
        prefetcher = prefetch.data_prefetcher(train_loader)
        images, target = prefetcher.next()
        # print(images.size())
        # print(target.size())
        i = 0
        optimizer_main_client.zero_grad()
        optimizer_proxy_clients.zero_grad()
        while images is not None:
            # adjust the lr first'
            scheduler_main_client(optimizer_main_client, i, current_round)
            scheduler_proxy_clients(optimizer_proxy_clients, i, current_round)
            i += 1
            fx,y_a, y_b, lam = model_main_client(images,
                                    target=target,
                                    mode='train',
                                    epoch=epoch,
                                    streams=streams)
            # main_client_fx = fx.clone().detach().requires_grad_(True)
            main_client_fx=[]
            for j in range(args.loop_factor):
                main_client_fx.append(fx[j].clone().detach().requires_grad_(True))
            ensemble_output, outputs, ce_loss, cot_loss = model_proxy_clients(main_client_fx,y_a, y_b, lam,target=target,
                                    mode='train',
                                    epoch=epoch,
                                    streams=streams)
            # measure accuracy and record loss
            batch_size_now = images.size(0)
            # notice the index i and j, avoid contradictory
            for j in range(args.loop_factor):
                acc1 = metric.accuracy(outputs[j, ...], target, topk=(1, ))
            total_iters = len(train_loader)
            # simply average outputs of small networks
            avg_acc1 = metric.accuracy(ensemble_output, target, topk=(1, ))

            avg_ce_loss = (ce_loss.mean().item(), batch_size_now)
            avg_cot_loss = (cot_loss.mean().item(), batch_size_now)
            # compute gradient and do SGD step
            total_loss = (ce_loss + cot_loss) / args.iters_to_accumulate

            dfx_client = []
            total_loss.backward()
            for j in range(args.loop_factor):
                dfx_client.append(main_client_fx[j].grad.clone().detach())
            for j in range(args.loop_factor):
                fx[j].backward(dfx_client[j])
            
            
            if i % args.iters_to_accumulate == 0 or i == total_iters:
                optimizer_main_client.step()
                optimizer_main_client.zero_grad()
                optimizer_proxy_clients.step()
                optimizer_proxy_clients.zero_grad()
            images, target = prefetcher.next()
    return total_loss.item()

def validate(val_loader, model_main_client,model_proxy_clients, args, streams=None):
   
    # switch to evaluate mode
    model_main_client.eval()
    model_proxy_clients.eval()
    with torch.no_grad():
        top1_all = []
        for i in range(args.loop_factor):
            top1_all.append(metric.AverageMeter('{}_Acc@1'.format(i), ':6.2f'))
        acc1_all = []
        acc5_all = []
        ce_loss_all = []
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute outputs and losses
           

            fx = model_main_client(images,
                                target=target,
                                mode='val',
                                )
            # main_client_fx = fx.clone().detach().requires_grad_(True)
            main_client_fx=[]
            for j in range(args.loop_factor):
                main_client_fx.append(fx[j].clone().detach().requires_grad_(True))
            ensemble_output, outputs, ce_loss = model_proxy_clients(main_client_fx,target=target,
                                    mode='val',
                                    )
            # ensemble_output, outputs, ce_loss = model(
            #     images, target=target, mode='val')

            # measure accuracy and record loss
            batch_size_now = images.size(0)
            for j in range(args.loop_factor):
                acc1, acc5 = metric.accuracy(
                    outputs[j, ...], target, topk=(1, 5))
                top1_all[j].update(acc1[0].item(), batch_size_now)

            # simply average outputs of small networks
            avg_acc1, avg_acc5 = metric.accuracy(
                ensemble_output, target, topk=(1, 5))
            acc1_all.append(avg_acc1)
            acc5_all.append(avg_acc5)
            ce_loss_all.append(ce_loss)
        avg_acc1_all = Average(acc1_all)
        avg_acc5_all = Average(acc5_all)
        avg_ce_loss_all = Average(ce_loss_all)
        
    # torch.cuda.empty_cache()
    return avg_ce_loss_all,avg_acc1_all,top1_all

def main(args):
    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely disable data parallelism.")
    # If we traing the model seperately, all the number of loops will be one.
        # It is similar as split_factor = 1
    args.loop_factor = 1 if args.is_train_sep or args.is_single_branch else args.split_factor
    # use distributed training or not
    args.is_distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function.
        # spawn will produce the process index for the first arg of main_worker
        torch.multiprocessing.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print("INFO:PyTorch: The number of GPUs in this node is {}".format(ngpus_per_node))

        # Simply call main_worker function
        print("INFO:PyTorch: Set gpu_id = 0 because you only have one visible gpu, otherwise, change the code")
        args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    args.model_dir = str(HOME)+"/models/splitnet/"+str(args.spid)
    if args.gpu is not None:
        if not args.evaluate:
            print("INFO:PyTorch: Use GPU: {} for training, the rank of this GPU is {}".format(
                args.gpu, args.rank))
        else:
            print("INFO:PyTorch: Use GPU: {} for evaluating, the rank of this GPU is {}".format(
                args.gpu, args.rank))
    setproctitle.setproctitle(args.proc_name + 'feddct_rank{}'.format(args.rank))
    if not args.multiprocessing_distributed or \
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        # define tensorboard summary
        val_writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'val'))

    if args.is_label_smoothing:
        criterion = label_smoothing.label_smoothing_CE(reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss()
    # create model
    if args.pretrained:
        model_info = "INFO:PyTorch: using pre-trained model '{}'".format(
            args.arch)
    else:
        model_info = "INFO:PyTorch: creating model '{}'".format(args.arch)
    print(model_info)
    global_model_main_client= splitnetsl.SplitNetMainClient(args,
                              norm_layer=norm.norm(args.norm_mode),
                              criterion=criterion)
    print("INFO:PyTorch: The number of parameters in the global main client model is {}".format(
        metric.get_the_number_of_params(global_model_main_client)))
     # print("INFO:PyTorch: The number of parameters in the model is {}".format(
    #     metric.get_the_number_of_params(model)))
    global_model_proxy_clients= splitnetsl.SplitNetProxyClient(args,
                              norm_layer=norm.norm(args.norm_mode),
                              criterion=criterion)
    print("INFO:PyTorch: The number of parameters in the global proxy clients model is {}".format(
        metric.get_the_number_of_params(global_model_proxy_clients)))
    
    models_main_client = [splitnetsl.SplitNetMainClient(args,
                              norm_layer=norm.norm(args.norm_mode),
                              criterion=criterion) for _ in range(args.num_selected)]

    models_proxy_clients = [splitnetsl.SplitNetProxyClient(args,
                              norm_layer=norm.norm(args.norm_mode),
                              criterion=criterion) for _ in range(args.num_selected)] 

    if not args.is_summary and not args.evaluate:
        save_hp_to_json(args)
    if args.is_summary:
        print(global_model_main_client)
        print(global_model_proxy_clients)
        return None
    summary.save_model_to_json(args, global_model_main_client)
    #summary.save_model_to_json(args, global_model_proxy_clients)
    
    global_model_main_client = global_model_main_client.cuda()
    global_model_proxy_clients = global_model_proxy_clients.cuda()

    for model in models_main_client:
        model = model.cuda()
    
    for model in models_proxy_clients:
        model = model.cuda()

    for model in models_main_client:
        # initial synchronizing with global model
        model.load_state_dict(global_model_main_client.state_dict())

    for model in models_proxy_clients:
        # initial synchronizing with global model
        model.load_state_dict(global_model_proxy_clients.state_dict())
    
    param_groups_main_clients = [model.parameters(
    ) if args.is_wd_all else lr_scheduler.get_parameter_groups(model) for model in models_main_client]

    param_groups_proxy_clients = [model.parameters(
    ) if args.is_wd_all else lr_scheduler.get_parameter_groups(model) for model in models_proxy_clients]

    if args.is_wd_all:
        print(
            "INFO:PyTorch: Applying weight decay to all learnable parameters in the model.")

    if args.optimizer == 'SGD':
        print("INFO:PyTorch: using SGD optimizer.")
        optimizers_main_clients = [torch.optim.SGD(param_group,
                                      args.lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay,
                                      nesterov=True if args.is_nesterov else False
                                      ) for param_group in param_groups_main_clients]
        optimizers_proxy_clients = [torch.optim.SGD(param_group,
                                      args.lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay,
                                      nesterov=True if args.is_nesterov else False
                                      ) for param_group in param_groups_proxy_clients]
    else:
        raise NotImplementedError
    
    # accelarate the training
    torch.backends.cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.resume:
        resume_proxy = (args.resume).replace("main_client","proxy_clients")
        if os.path.isfile(args.resume) and os.path.isfile(resume_proxy):
            print("INFO:PyTorch: => loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint_main_client = torch.load(args.resume)
                checkpoint_proxy_clients = torch.load(resume_proxy)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint_main_client = torch.load(args.resume, map_location=loc)
                checkpoint_proxy_clients = torch.load(resume_proxy, map_location=loc)
            args.start_round = checkpoint_proxy_clients['round']
            best_acc1 = checkpoint_proxy_clients['best_acc1']
            print("INFO:PyTorch: Loading state_dict of main client")
            global_model_main_client.load_state_dict(checkpoint_main_client['state_dict'])
            for optimizer in optimizers_main_clients:
                optimizer.load_state_dict(checkpoint_main_client['optimizer'])
            print("INFO:PyTorch: Loading state_dict of proxy clients")
            global_model_proxy_clients.load_state_dict(checkpoint_proxy_clients['state_dict'])
            for optimizer in optimizers_main_clients:
                optimizer.load_state_dict(checkpoint_main_client['optimizer'])
            print("INFO:PyTorch: => loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint_main_client['round']))
            print("INFO:PyTorch: => loaded checkpoint '{}' (epoch {})".format(
                resume_proxy, checkpoint_proxy_clients['round']))
            for model in models_main_client:
                # initial synchronizing with global model
                model.load_state_dict(global_model_main_client.state_dict())
            for model in models_proxy_clients:
                # initial synchronizing with global model
                model.load_state_dict(global_model_proxy_clients.state_dict())
        else:
            print("INFO:PyTorch: => no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_round  = 0 
    # Data loading code
    data_split_factor = args.loop_factor if args.is_diff_data_train else 1
    print("INFO:PyTorch: => The number of views of train data is '{}'".format(
        data_split_factor))
    
    train_loader, train_sampler = factory.get_data_loader(args.data,
                                                          split_factor=data_split_factor,
                                                          batch_size=args.batch_size,
                                                          crop_size=args.crop_size,
                                                          dataset=args.dataset,
                                                          split="train",
                                                          is_distributed=args.is_distributed,
                                                          is_autoaugment=args.is_autoaugment,
                                                          randaa=args.randaa,
                                                          is_cutout=args.is_cutout,
                                                          erase_p=args.erase_p,
                                                          num_workers=args.workers,
                                                          is_fed=args.is_fed,
                                                          num_clusters=args.num_clusters,
                                                          cifar10_non_iid = args.cifar10_non_iid,
                                                          cifar100_non_iid= args.cifar100_non_iid)

    val_loader = factory.get_data_loader(args.data,
                                         batch_size=args.eval_batch_size,
                                         crop_size=args.crop_size,
                                         dataset=args.dataset,
                                         split="val",
                                         num_workers=args.workers,
                                         cifar10_non_iid = args.cifar10_non_iid,
                                         cifar100_non_iid=args.cifar100_non_iid)

    print(train_loader)
    # learning rate scheduler
    # schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.num_rounds) for optimizer in optimizers]

	# schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99) for optimizer in optimizers]
    schedulers_main_clients = [lr_scheduler.lr_scheduler(mode=args.lr_mode,
                                          init_lr=args.lr,
                                          num_epochs=args.num_rounds,
                                          iters_per_epoch=len(train_loader),
                                          lr_milestones=args.lr_milestones,
                                          lr_step_multiplier=args.lr_step_multiplier,
                                          slow_start_epochs=args.slow_start_epochs,
                                          slow_start_lr=args.slow_start_lr,
                                          end_lr=args.end_lr,
                                          multiplier=args.lr_multiplier,
                                          decay_factor=args.decay_factor,
                                          decay_epochs=args.decay_epochs,
                                          staircase=True
                                          )for optimizer in optimizers_main_clients]
    
    schedulers_proxy_clients = [lr_scheduler.lr_scheduler(mode=args.lr_mode,
                                          init_lr=args.lr,
                                          num_epochs=args.num_rounds,
                                          iters_per_epoch=len(train_loader),
                                          lr_milestones=args.lr_milestones,
                                          lr_step_multiplier=args.lr_step_multiplier,
                                          slow_start_epochs=args.slow_start_epochs,
                                          slow_start_lr=args.slow_start_lr,
                                          end_lr=args.end_lr,
                                          multiplier=args.lr_multiplier,
                                          decay_factor=args.decay_factor,
                                          decay_epochs=args.decay_epochs,
                                          staircase=True
                                          )for optimizer in optimizers_proxy_clients]
    
    streams = None
    saved_ckpt_filenames = []
    saved_ckpt_filenames_ = []

    # Runnining FedDCT
    for r in range(args.start_round, args.num_rounds + 1):
        print("\nusing random cluster")
        client_idx = np.random.permutation(args.num_clusters*args.loop_factor)[
            :(args.num_selected*args.loop_factor)]
        # client update'
        client_cluster_idx = np.split(client_idx, args.num_selected)
        print("\nrandom cluster is "+ str(client_cluster_idx))
        loss = 0
        for i in tqdm(range(args.num_selected)): # number of clusters chosen perround
            
            current_client_idx = client_cluster_idx[i] # show clients in the cluster
            print("\nCurrent cluster is "+ str(current_client_idx))
            for j in tqdm(range(args.loop_factor)): # fpr each client in the cluster
            # client_update(args,client_model,scheduler,optimizer,train_loader,epoch=5,streams=None,scaler=None)
                print("\nCurrent client is "+ str(current_client_idx[j]))
                loss += client_update(args,r, models_main_client[i],models_proxy_clients[i], schedulers_main_clients[i],schedulers_proxy_clients[i],optimizers_main_clients[i],optimizers_proxy_clients[i],
                                    train_loader[current_client_idx[j]], epoch=args.fed_epochs, streams=streams) # train model with data from each client
                val_writer.add_scalar(
                    'learning_rate_'+str(i), optimizers_main_clients[i].param_groups[0]['lr'], global_step=r)
                val_writer.add_scalar(
                    'learning_rate_proxy'+str(i), optimizers_proxy_clients[i].param_groups[0]['lr'], global_step=r)
        # model aggregate. In the paper the main client model aggregation and proxy model 
        # aggregation runs seperatedly, but for convenience sake, we put it into one function 
        server_aggregate(global_model_main_client,global_model_proxy_clients,models_main_client, models_proxy_clients)
        test_loss, acc,top1_all = validate(val_loader, global_model_main_client,global_model_proxy_clients, args)
        is_best = acc > best_acc1
        best_acc1 = max(acc, best_acc1)
        val_writer.add_scalar(
            'average training loss', (loss / args.num_selected), global_step=r)
        val_writer.add_scalar(
            'test loss', test_loss.cpu().data.numpy(), global_step=r)
        val_writer.add_scalar(
            'test acc', acc.cpu().data.numpy(), global_step=r)
        val_writer.add_scalar(
            'best_acc1', best_acc1.cpu().data.numpy(), global_step=r)
        for sl in range(args.loop_factor):
                    val_writer.add_scalar('{}_acc1'.format(
                        sl), top1_all[sl].avg, global_step=r)
        # for i in tqdm(range(args.num_selected)):
        #     val_writer.add_scalar('best_acc1',top1_all[i].avg, global_step=r)
        # save checkpoints
        # filename = "main_client_checkpoint_{0}.pth.tar".format(r)
        # filename_ = "proxy_clients_checkpoint_{0}.pth.tar".format(r)
        # saved_ckpt_filenames.append(filename)
        # saved_ckpt_filenames_.append(filename_)
        # # remove the oldest file if the number of saved ckpts is greater than args.max_ckpt_nums
        # if len(saved_ckpt_filenames) > args.max_ckpt_nums:
        #     os.remove(os.path.join(args.model_dir,
        #                                    saved_ckpt_filenames.pop(0)))
        # if len(saved_ckpt_filenames_) > args.max_ckpt_nums:
        #     os.remove(os.path.join(args.model_dir,
        #                                    saved_ckpt_filenames_.pop(0)))
        ckpt_dict_main_client = {
            'round': r + 1,
            'arch': args.arch,
            'state_dict': global_model_main_client.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizers_main_clients[i].state_dict(),
        }

        ckpt_dict_proxy_clients = {
            'round': r + 1,
            'arch': args.arch,
            'state_dict': global_model_proxy_clients.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizers_proxy_clients[i].state_dict(),
        }

        # metric.save_checkpoint(
        #         ckpt_dict_main_client, is_best, args.model_dir, filename=filename)
        # metric.save_checkpoint(
        #         ckpt_dict_proxy_clients, is_best, args.model_dir, filename=filename_)
        
        metric.save_checkpoint_main_client(
                ckpt_dict_main_client, is_best, args.model_dir)
        metric.save_checkpoint_proxy_clients(
                ckpt_dict_proxy_clients, is_best, args.model_dir)
        print('%d-th round' % r)
        print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' %
              (loss / args.num_selected, test_loss, acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
    args = train_params.add_parser_params(parser)
    assert args.is_fed == 1, "For fed learning, args.if_fed must be 1"
    os.makedirs(args.model_dir, exist_ok=True)
    print(args)
    main(args)