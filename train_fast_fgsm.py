import argparse
import logging
import sys  # 导入系统库
import os
import time
import math  # 导入数学库

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preactresnet import PreActResNet18
from utils_free_fgsm import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard)

logger = logging.getLogger(__name__)

class MyArgs:
    batch_size = 128
    data_dir = '../../cifar-data'
    epochs = 200
    lr_schedule = 'cyclic'
    lr_min = 0.0
    lr_max = 0.04
    weight_decay = 5e-4
    momentum = 0.9
    epsilon = 8
    minibatch_replays = 8
    out_dir = 'train_free_output'
    seed = 0
    opt_level = 'O2'
    loss_scale = '1.0'
    master_weights = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int, help='Total number of epochs will be this argument * number of minibatch replays.')
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.04, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--minibatch-replays', default=8, type=int)
    parser.add_argument('--out-dir', default='train_free_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    print(args)  # Changed from logger.info to print

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std

    model = PreActResNet18().cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.opt_level == 'O2' and args.master_weights:
        # Additional handling if needed for master_weights
        pass
    criterion = nn.CrossEntropyLoss()

    delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    delta.requires_grad = True

    lr_steps = args.epochs * len(train_loader) * args.minibatch_replays
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    start_train_time = time.time()
    print('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')  # Changed from logger.info to print
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            for _ in range(args.minibatch_replays):
                output = model(X + delta[:X.size(0)])
                loss = criterion(output, y)
                opt.zero_grad()
                loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + epsilon * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                opt.step()
                delta.grad.zero_()
                scheduler.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        print(f'{epoch} \t {epoch_time - start_epoch_time:.1f} \t {lr:.4f} \t {train_loss/train_n:.4f} \t {train_acc/train_n:.4f}')  # Changed from logger.info to print
    train_time = time.time()
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    print(f'Total train time: {(train_time - start_train_time)/60:.4f} minutes')  # Changed from logger.info to print

    # Evaluation
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    print(f'Test Loss \t Test Acc \t PGD Loss \t PGD Acc')  # Changed from logger.info to print
    print(f'{test_loss:.4f} \t {test_acc:.4f} \t {pgd_loss:.4f} \t {pgd_acc:.4f}')  # Changed from logger.info to print


if __name__ == "__main__":
    main()