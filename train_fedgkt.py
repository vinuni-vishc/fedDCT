import argparse
import os
import torch
import warnings
from dataset import factory
from params import train_params
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.model.cv.resnet56_gkt.resnet_client import resnet8_56
from fedml_api.model.cv.resnet56_gkt.resnet_pretrained import resnet56_pretrained
from fedml_api.model.cv.resnet56_gkt.resnet_server import resnet56_server
from fedml_api.model.cv.resnet_gkt.resnet import wide_resnet16_8_gkt,wide_resnet50_2_gkt,resnet110_gkt
from fedml_api.distributed.fedgkt.GKTClientTrainer import GKTClientTrainer
from fedml_api.distributed.fedgkt.GKTServerTrainer import GKTServerTrainer
from params.train_params import save_hp_to_json
from config import HOME
from tensorboardX import SummaryWriter
device = torch.device("cuda:0")
def main(args):
    # Data loading code
    args.model_dir = str(HOME)+"/models/splitnet/"+str(args.spid)
    if not args.is_summary and not args.evaluate:
        save_hp_to_json(args)
    val_writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'val'))
    args.loop_factor = 1 if args.is_train_sep or args.is_single_branch else args.split_factor
    data_split_factor = args.loop_factor if args.is_diff_data_train else 1
    args.is_distributed = args.world_size > 1 or args.multiprocessing_distributed
    print("INFO:PyTorch: => The number of views of train data is '{}'".format(
        data_split_factor))
    train_data_local_dict, train_sampler = factory.get_data_loader(args.data,
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
    test_data_global = factory.get_data_loader(args.data,
                                         batch_size=args.eval_batch_size,
                                         crop_size=args.crop_size,
                                         dataset=args.dataset,
                                         split="val",
                                         num_workers=args.workers,
                                         cifar10_non_iid = args.cifar10_non_iid,
                                         cifar100_non_iid= args.cifar100_non_iid)
    if args.dataset == "cifar10":
        data_loader = load_partition_data_cifar10
    elif args.dataset == "cifar100":
        data_loader = load_partition_data_cifar100
    client_number = args.num_clusters*args.split_factor
    train_data_num, test_data_num, train_data_global, _, \
    _, _, test_data_local_dict, \
    class_num = data_loader(args.dataset, args.data, 'homo',
                            0.5, client_number, args.batch_size)
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
            train_data_local_dict, test_data_local_dict, class_num]
    model_client,model_server = wide_resnet16_8_gkt()
    print("server and client create complete")
    round_idx = 0
    client_trainer = []
    for i in range (0,client_number):
 
        client_trainer.append(GKTClientTrainer(i, train_data_local_dict, test_data_local_dict,
                                        device, model_client, args))
    server_trainer = GKTServerTrainer(client_number, device, model_server, args,val_writer)
    for round in range (0,args.num_rounds):
        for i in range (0,client_number):
            extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test, labels_dict_test = client_trainer[i].train()
            print("finish training client")
            server_trainer.add_local_trained_result(i, extracted_feature_dict, logits_dict, labels_dict,
                                                            extracted_feature_dict_test, labels_dict_test)
        b_all_received = server_trainer.check_whether_all_receive()
        if b_all_received:
            print("server received all")
            server_trainer.train(round_idx)
            print("server finished training")
            round_idx += 1
        for i in range (0,client_number):
            global_logits = server_trainer.get_global_logits(i)
            client_trainer[i].update_large_model_logits(global_logits)
        print("sent back to client")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = train_params.add_parser_params(parser)
    assert args.is_fed == 1, "For fed learning, args.if_fed must be 1"

    os.makedirs(args.model_dir, exist_ok=True)
    print(args)
    main(args)
