import argparse  # 导入命令行参数解析库
import logging  # 导入日志库
import sys  # 导入系统库
import time  # 导入时间库
import math  # 导入数学库

import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数库
from torch.autograd import Variable  # 导入自动求导变量

import os  # 导入操作系统库

from wideresnet import WideResNet  # 导入自定义的WideResNet模块
from preactresnet import PreActResNet18  # 导入自定义的PreActResNet18模块

from utils import *  # 导入自定义的实用函数

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()  # 设置CIFAR-10数据集的均值
std = torch.tensor(cifar10_std).view(3,1,1).cuda()  # 设置CIFAR-10数据集的标准差

def normalize(X):  # 定义标准化函数
    return (X - mu)/std  # 返回标准化后的张量

upper_limit, lower_limit = 1,0  # 设置张量值的上下限

def clamp(X, lower_limit, upper_limit):  # 定义张量值限制函数
    return torch.max(torch.min(X, upper_limit), lower_limit)  # 返回限制后的张量


class Batches():  # 定义一个名为Batches的类，用于封装PyTorch的DataLoader
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):  # 初始化函数
        self.dataset = dataset  # 将传入的数据集赋值给self.dataset
        self.batch_size = batch_size  # 将传入的批处理大小赋值给self.batch_size
        self.set_random_choices = set_random_choices  # 将传入的set_random_choices赋值给self.set_random_choices
        self.dataloader = torch.utils.data.DataLoader(  # 创建一个PyTorch数据加载器
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):  # 定义迭代器函数
        if self.set_random_choices:  # 如果set_random_choices为True
            self.dataset.set_random_choices()  # 调用数据集的set_random_choices方法
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)  # 返回一个生成器，用于迭代数据

    def __len__(self):  # 定义获取长度的函数
        return len(self.dataloader)  # 返回数据加载器的长度


def mixup_data(x, y, alpha=1.0):  # 定义名为mixup_data的函数，接受输入x, y和alpha，用于实现Mixup数据增强技术
    '''Returns mixed inputs, pairs of targets, and lambda'''  # 函数文档字符串，描述函数的功能
    if alpha > 0:  # 判断alpha是否大于0
        lam = np.random.beta(alpha, alpha)  # 如果是，从Beta分布中随机生成一个lambda值
    else:
        lam = 1  # 否则，设置lambda为1

    batch_size = x.size()[0]  # 获取输入x的批量大小
    index = torch.randperm(batch_size).cuda()  # 随机生成一个索引排列并移至GPU

    mixed_x = lam * x + (1 - lam) * x[index, :]  # 计算混合后的输入
    y_a, y_b = y, y[index]  # 获取原始和排列后的目标标签

    return mixed_x, y_a, y_b, lam  # 返回混合后的输入、目标标签对和lambda值


def mixup_criterion(criterion, pred, y_a, y_b, lam):  # 定义名为mixup_criterion的函数，接受损失函数、预测值、两组标签和lambda值，用于计算使用Mixup数据增强技术后的损失值
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)  # 返回混合后的损失值


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):  # 定义名为attack_pgd的函数，用于执行投影梯度下降（PGD）攻击
    max_loss = torch.zeros(y.shape[0]).cuda()  # 初始化最大损失为0
    max_delta = torch.zeros_like(X).cuda()  # 初始化最大扰动为0
    for _ in range(restarts):  # 进行多次重启以找到最有效的攻击
        delta = torch.zeros_like(X).cuda()  # 初始化扰动为0
        # 根据范数类型初始化扰动
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError  # 如果范数类型未知，则抛出错误
        delta = clamp(delta, lower_limit-X, upper_limit-X)  # 限制扰动在允许的范围内
        delta.requires_grad = True  # 设置delta为需要梯度

        # 执行多次梯度下降步骤以找到有效的扰动
        for _ in range(attack_iters):
            output = model(normalize(X + delta))  # 计算模型输出
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]  # 如果早停启用，找出模型预测正确的样本索引
            else:
                index = slice(None,None,None)  # 否则，使用一个全切片，表示所有样本
            if not isinstance(index, slice) and len(index) == 0:
                break  # 如果没有模型预测正确的样本，并且早停启用，则退出循环
            if mixup:
                criterion = nn.CrossEntropyLoss()  # 如果启用Mixup，设置交叉熵损失函数
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)  # 计算Mixup损失
            else:
                loss = F.cross_entropy(output, y)  # 否则，计算普通的交叉熵损失
            loss.backward()  # 反向传播计算梯度
            grad = delta.grad.detach()  # 获取扰动的梯度

            # 更新扰动
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]

            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()  # 清零梯度
        
        # 计算使用该扰动的损失
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')

        # 更新最大损失和对应的扰动
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta  # 返回找到的最有效扰动


def get_args():  # 定义一个名为get_args的函数，用于获取命令行参数
    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
    parser.add_argument('--model', default='PreActResNet18')  # 指定模型类型，默认为'PreActResNet18'
    parser.add_argument('--l2', default=0, type=float)  # L2正则化系数，默认为0
    parser.add_argument('--l1', default=0, type=float)  # L1正则化系数，默认为0
    parser.add_argument('--batch-size', default=128, type=int)  # 批处理大小，默认为128
    parser.add_argument('--data-dir', default='../cifar-data', type=str)  # 数据目录，默认为'../cifar-data'
    parser.add_argument('--epochs', default=200, type=int)  # 训练周期数，默认为200
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine'])  # 学习率调度策略，默认为'piecewise'
    parser.add_argument('--lr-max', default=0.1, type=float)  # 最大学习率，默认为0.1
    parser.add_argument('--lr-one-drop', default=0.01, type=float)  # 单次下降的学习率，默认为0.01
    parser.add_argument('--lr-drop-epoch', default=100, type=int)  # 学习率下降的周期，默认为100
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])  # 攻击类型，默认为'pgd'
    parser.add_argument('--epsilon', default=8, type=int)  # 攻击强度（epsilon值），默认为8
    parser.add_argument('--attack-iters', default=10, type=int)  # 攻击强度（epsilon值），默认为8
    parser.add_argument('--restarts', default=1, type=int)  # 攻击重启次数，默认为1
    parser.add_argument('--pgd-alpha', default=2, type=float)  # PGD攻击的alpha值，默认为2
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)  # FGSM攻击的alpha值，默认为1.25
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])  # 范数类型，默认为'l_inf'
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])  # FGSM的初始化策略，默认为'random'
    parser.add_argument('--fname', default='cifar_model', type=str)  # 输出模型的文件名，默认为'cifar_model'
    parser.add_argument('--seed', default=0, type=int)  # 随机种子，默认为0
    parser.add_argument('--half', action='store_true')  # 是否使用半精度浮点数，这是一个标志参数
    parser.add_argument('--width-factor', default=10, type=int)  # 模型宽度因子，默认为10
    parser.add_argument('--resume', default=0, type=int)  # 是否从先前的检查点恢复，默认为0（不恢复）
    parser.add_argument('--cutout', action='store_true')  # 是否使用Cutout数据增强，这是一个标志参数
    parser.add_argument('--cutout-len', type=int)  # Cutout的长度，如果提供
    parser.add_argument('--mixup', action='store_true')  # 是否使用Mixup数据增强，这是一个标志参数
    parser.add_argument('--mixup-alpha', type=float)  # Mixup的alpha值，如果提供
    parser.add_argument('--eval', action='store_true')  # 是否处于评估模式，这是一个标志参数
    parser.add_argument('--val', action='store_true')  #是否使用验证集，这是一个标志参数
    parser.add_argument('--chkpt-iters', default=10, type=int)  # 检查点保存的迭代间隔，默认为10
    return parser.parse_args()  # 解析命令行参数并返回


def main():
    args = get_args()  # 获取命令行参数

    if not os.path.exists(args.fname):  # 检查输出目录是否存在
        os.makedirs(args.fname)

    # 初始化日志记录器
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)  # 记录命令行参数

    # 设置随机种子以保证可重复性
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 初始化数据转换和数据集
    transforms = [Crop(32, 32), FlipLR()]
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))
    if args.val:
        try:
            dataset = torch.load("cifar10_validation_split.pth")
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
            return
        val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=2)
    else:
        dataset = cifar10(args.data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=2)

    # 设置攻击参数
    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    # 初始化模型
    if args.model == 'PreActResNet18':
        model = PreActResNet18()
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()  # 使用数据并行处理
    model.train()  # 设置模型为训练模式

    # 初始化优化器和损失函数
    if args.l2:
        decay, no_decay = [], []
        for name,param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model.parameters()

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    # 如果使用 'free' 或 'fgsm' 攻击，初始化攻击变量
    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    if args.attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs
    
    # 初始化学习率计划
    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine': 
        def lr_schedule(t): 
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))

    # 初始化最佳准确度变量
    best_test_robust_acc = 0
    best_val_robust_acc = 0
    # 从检查点恢复
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(args.fname, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        best_test_robust_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
        if args.val:
            best_val_robust_acc = torch.load(os.path.join(args.fname, f'model_val.pth'))['val_robust_acc']
    else:
        start_epoch = 0
    
    # 主训练和评估循环
    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    
    # 主训练和评估循环开始
    for epoch in range(start_epoch, epochs):
        model.train()  # 设置模型为训练模式
        start_time = time.time()  # 记录训练开始时间

        # 初始化训练统计变量
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0

        # 训练批次循环
        for i, batch in enumerate(train_batches):
            if args.eval:  # 如果是评估模式，跳出循环
                break
            X, y = batch['input'], batch['target']  # 获取输入和标签

            # 如果使用Mixup数据增强
            if args.mixup:
                X, y_a, y_b, lam = mixup_data(X, y, args.mixup_alpha)
                X, y_a, y_b = map(Variable, (X, y_a, y_b))
            
            # 更新学习率
            lr = lr_schedule(epoch + (i + 1) / len(train_batches))
            opt.param_groups[0].update(lr=lr)

            # 执行攻击（如果有）
            if args.attack == 'pgd':
                # Random initialization
                if args.mixup:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                else:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                delta = delta.detach()
            elif args.attack == 'fgsm':
                delta = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
            # Standard training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)

            # 计算模型输出和损失
            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            if args.mixup:
                robust_loss = mixup_criterion(criterion, robust_output, y_a, y_b, lam)
            else:
                robust_loss = criterion(robust_output, y)

            if args.l1:
                for name,param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1*param.abs().sum()

            # 更新优化器和模型权重
            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            # 更新训练统计变量
            output = model(normalize(X))
            if args.mixup:
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                loss = criterion(output, y)

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        # 记录训练结束时间
        train_time = time.time()

        model.eval()# 设置模型为评估模式

        # 初始化测试统计变量
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        
        # 测试批次循环
        for i, batch in enumerate(test_batches):
            X, y = batch['input'], batch['target']

            # Random initialization
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, early_stop=args.eval)
            delta = delta.detach()

            # 计算模型输出和损失
            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss = criterion(robust_output, y)

            output = model(normalize(X))
            loss = criterion(output, y)

            # 更新测试统计变量
            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)

        test_time = time.time()  # 记录测试结束时间

        if args.val:
            val_loss = 0
            val_acc = 0
            val_robust_loss = 0
            val_robust_acc = 0
            val_n = 0
            for i, batch in enumerate(val_batches):
                X, y = batch['input'], batch['target']

                # Random initialization
                if args.attack == 'none':
                    delta = torch.zeros_like(X)
                else:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, early_stop=args.eval)
                delta = delta.detach()

                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                robust_loss = criterion(robust_output, y)

                output = model(normalize(X))
                loss = criterion(output, y)

                val_robust_loss += robust_loss.item() * y.size(0)
                val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                val_loss += loss.item() * y.size(0)
                val_acc += (output.max(1)[1] == y).sum().item()
                val_n += y.size(0)

        if not args.eval:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, lr,
                train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)

            if args.val:
                logger.info('validation %.4f \t %.4f \t %.4f \t %.4f',
                    val_loss/val_n, val_acc/val_n, val_robust_loss/val_n, val_robust_acc/val_n)

                if val_robust_acc/val_n > best_val_robust_acc:
                    torch.save({
                            'state_dict':model.state_dict(),
                            'test_robust_acc':test_robust_acc/test_n,
                            'test_robust_loss':test_robust_loss/test_n,
                            'test_loss':test_loss/test_n,
                            'test_acc':test_acc/test_n,
                            'val_robust_acc':val_robust_acc/val_n,
                            'val_robust_loss':val_robust_loss/val_n,
                            'val_loss':val_loss/val_n,
                            'val_acc':val_acc/val_n,
                        }, os.path.join(args.fname, f'model_val.pth'))
                    best_val_robust_acc = val_robust_acc/val_n

            # save checkpoint
            if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
                torch.save(model.state_dict(), os.path.join(args.fname, f'model_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))

            # save best
            if test_robust_acc/test_n > best_test_robust_acc:
                torch.save({
                        'state_dict':model.state_dict(),
                        'test_robust_acc':test_robust_acc/test_n,
                        'test_robust_loss':test_robust_loss/test_n,
                        'test_loss':test_loss/test_n,
                        'test_acc':test_acc/test_n,
                    }, os.path.join(args.fname, f'model_best.pth'))
                best_test_robust_acc = test_robust_acc/test_n
        else:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, -1,
                -1, -1, -1, -1,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
            return


if __name__ == "__main__":
    main()
