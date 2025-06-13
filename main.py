import os
import torch
import argparse
import gc
from model.configs import Config, str2bool
from torch.utils.data import DataLoader
from model.mrhisum_dataset import MrHiSumDataset, BatchCollator
from model.solver import Solver
import random
import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


set_seed(42)

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    g = torch.Generator()
    g.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP',
                        help='the name of the model')
    parser.add_argument('--epochs', type=int, default=200,
                        help='the number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='the learning rate')
    parser.add_argument('--l2_reg', type=float,
                        default=1e-4, help='l2 regularizer')
    parser.add_argument('--dropout_ratio', type=float,
                        default=0.5, help='the dropout ratio')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='the batch size')
    parser.add_argument('--tag', type=str, default='dev',
                        help='A tag for experiments')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='checkpoint path for inference or weight initialization')
    parser.add_argument('--train', type=str2bool,
                        default='true', help='when use Train')
    parser.add_argument('--path', type=str,
                        default='dataset/metadata_split.json', help='path')
    parser.add_argument('--device', type=str, default='0', help='gpu')
    parser.add_argument('--modal', type=str,
                        default='visual', help='visual,audio,multi')
    parser.add_argument('--beta', type=float, default=0, help='beta')
    parser.add_argument('--type', type=str, default='base',
                        help='base,ib,cib,eib,lib')  # cib,eib,lib

    opt = parser.parse_args()
    # print(type(opt))
    # print(opt)
    kwargs = vars(opt)
    config = Config(**kwargs)
    # print(config)
    # print(type(config))
    # os._exit()
    train_dataset = MrHiSumDataset(mode='train', path=config.path)
    val_dataset = MrHiSumDataset(mode='val', path=config.path)
    test_dataset = MrHiSumDataset(mode='test', path=config.path)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=4, collate_fn=BatchCollator(), worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=4)
    device = torch.device("cuda:"+config.device)
    print("Device being used:", device)
    solver = Solver(config, train_loader, val_loader,
                    test_loader, device, config.modal)

    solver.build()
    test_model_ckpt_path = None
    # proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if config.train:
        # best_path, best_f1_ckpt_path, best_map50_ckpt_path, best_map15_ckpt_path = solver.train()
        best_path, best_path50, best_path100, best_path150, best_path200, = solver.train()
        # print(best_path, best_path50, best_path100)
        for eachpath in best_path:
            solver.test(eachpath)
        for eachpath in best_path50:
            solver.test(eachpath)
        for eachpath in best_path100:
            solver.test(eachpath)
        for eachpath in best_path150:
            solver.test(eachpath)
        for eachpath in best_path200:
            solver.test(eachpath)
        # solver.test(best_f1_ckpt_path)
        # solver.test(best_map50_ckpt_path)
        # solver.test(best_map15_ckpt_path)
        # solver.test(best_pre_ckpt_path)
        print(best_path, best_path50, best_path100)
    else:
        test_model_ckpt_path = config.ckpt_path
        if test_model_ckpt_path == None:
            print("Trained model checkpoint requried. Exit program")
            exit()
        else:
            solver.test(test_model_ckpt_path)
