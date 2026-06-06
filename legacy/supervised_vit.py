import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import copy
from data.sub_dataset import Sub_Dataset
from util.utils import *
from util.datasets import build_dataset, build_transform
import model.vit as vit
import util.lr_decay as lrd
import util.lr_sched as lr_sched


# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,6,7'


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('--nb_classes', default=1000, type=int,
                    help='number of the classification types')
parser.add_argument('--output_dir', default = '', type = str)
parser.add_argument('--log_dir', default='./output_dir',
                    help='path where to tensorboard log')
parser.add_argument('--data_path', default = '/home/ljl/Datasets/ImageNet/',
                    help='path to dataset')
parser.add_argument('--sub_batch_size',default = 256, type=int, metavar='N',
                    help='mini-batch size of sub dataloader')
parser.add_argument('--batch_size', default = 256, type=int,
                    help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
parser.add_argument('--accum_iter', default=1, type=int,
                    help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

# Model parameters
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_small_patch16')

parser.add_argument('--epochs', default = 300, type=int)

parser.add_argument('--input_size', default=224, type=int,
                    help='images input size')

parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

# Optimizer parameters
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--weight_decay', type=float, default=0.3,
                    help='weight decay (default: 0.05)')

parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate (absolute lr)')
parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--layer_decay', type=float, default=0.75,
                    help='layer-wise lr decay from ELECTRA/BEiT')

parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')

parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                    help='epochs to warmup LR')

# Augmentation parameters
parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                    help='Color jitter factor (enabled only when not using Auto/RandAug)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')

# * Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

# * Finetuning params
parser.add_argument('--finetune', default='',
                    help='finetune from checkpoint')
parser.add_argument('--global_pool', action='store_true')
parser.set_defaults(global_pool=True)
parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                    help='Use class token instead of global pool for classification')


parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--resume', default='',
                    help='resume from checkpoint')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--evaluate', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--dist_eval', action='store_true', default=False,
                    help='Enabling distributed evaluation (recommended during training for faster monitor')
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
parser.set_defaults(pin_mem=True)


parser.add_argument('-j', '--workers', default = 10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('-b', '--batch-size', default = 256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')


parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = vit.__dict__[args.arch](
            num_classes=1000,
            drop_path_rate = args.drop_path,
            global_pool = args.global_pool,
        )
    model_without_ddp = model
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.sub_batch_size = int(args.sub_batch_size / ngpus_per_node)

            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # Data loading code
    train_dataset = build_dataset(is_train=True, args = args)
    val_dataset = build_dataset(is_train=False, args=args)
    transform_train = build_transform(is_train=False, args=args)

    # 用来生成评估loss contribution的batches
    sub_dataset = Sub_Dataset(
        full_dataset = train_dataset,
        imagenet_root = os.path.join(args.data_path, 'train'),
        num_classes = 1000,
        images_per_class = 12,
        transform = transform_train,
        )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        sub_sampler = torch.utils.data.distributed.DistributedSampler(sub_dataset)
    else:
        train_sampler = None
        eval_sampler = None
        sub_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    sub_dataloader = torch.utils.data.DataLoader(
        sub_dataset, batch_size=args.sub_batch_size, shuffle=(sub_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=sub_sampler)
    
    batches = []
    for inputs, targets in sub_dataloader:
        batches.append((inputs, targets))
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(eval_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler = eval_sampler)
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256
    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas = (0.9, 0.95))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy().cuda(args.gpu)
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda(args.gpu)
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

    print("criterion = %s" % str(criterion))

    args.iter = 0
    effe_direction = None
    tau_1 = 10
    tau_2 = 5
    n = 19
    ema_factor = (n-1)/(n+1)
    path_contributions = []
    loss_contributions = []
    contributions = []
    accumulated_importance = None
    accumulated_path_integral = None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if 'path_contribution' in checkpoint:
                path_contributions = checkpoint['path_contribution']
            if 'loss_contribution' in checkpoint:
                loss_contributions = checkpoint['loss_contribution']
            if 'contribution' in checkpoint:
                contribution = checkpoint['contribution']
            # if 'accumulated_importance' in checkpoint:
            #     accumulated_importance = checkpoint['accumulated_importance']
            if 'effe_direction' in checkpoint:
                effe_direction = checkpoint['effe_direction']
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch =====================================
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # we use a per iteration (instead of per epoch) lr scheduler
            if i % args.accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)
            
            ######### loss before=================================================
            sub_inputs, sub_targets = batches[args.iter % len(batches)]
            if args.gpu is not None:
                sub_inputs = sub_inputs.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                sub_targets = sub_targets.cuda(args.gpu, non_blocking=True)
            if mixup_fn is not None:
                sub_inputs, sub_targets = mixup_fn(sub_inputs, sub_targets)
            loss_before = comp_loss_batch(args, model, sub_inputs, sub_targets, criterion)
            # ====================================================================

            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            if mixup_fn is not None:
                images, target = mixup_fn(images, target)
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss

            acc1, acc5 = accuracy(output, torch.argmax(target, dim=1), topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()


            # ============================================================
            model_copy = copy.deepcopy(model)
            param_before_update = get_weights(model_copy)
            # ============================================================

            optimizer.step()

            # ============================================================
            delta_param = add_list_tensor(get_weights(model),param_before_update,-1)
            gradients = [param.grad.clone().detach() for param in model.parameters() if param.grad is not None]
            path_integral = compute_importance(model_copy, gradients, delta_param, images, target, criterion)

            if accumulated_path_integral is None:
                accumulated_path_integral = path_integral
            else:
                accumulated_path_integral = add_list_tensor(accumulated_path_integral, path_integral)
                
            # # 计算 loss_contribution
            loss_after = comp_loss_batch(args, model, sub_inputs, sub_targets, criterion)

            loss_contribution = loss_before - loss_after
            loss_contributions.append(loss_contribution)

            # # 计算 path_contribution
            if effe_direction is None:
                effe_direction = [d.detach().clone() for d in delta_param]
            
            path_contribution = get_projection_length(delta_param, effe_direction)
            path_contributions.append(path_contribution)

            # # 更新effe_direction
            if args.iter != 0: 
                effe_direction = update_ema_direction(effe_direction, delta_param, ema_factor)

            # # # 计算 contribution
            # contribution = math.pow((1 + tau_1 * path_contribution), (1+ tau_2 * loss_contribution))
            # contributions.append(contribution)

            
            # importance = rescale(importance, contribution)
            importance = rescale_importance_by_contribution(path_integral, loss_contribution, path_contribution, tau_1, tau_2)
            importance = checknan(importance)
            
            if accumulated_importance is None:
                accumulated_importance = importance
            else:
                accumulated_importance = add_list_tensor(accumulated_importance, importance)
            args.iter += 1
            # ==============================================================

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        
        
        torch.cuda.synchronize()
        # ======================== training status =================================
        save_dict = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'effe_direction' : effe_direction,
                'path_contribution': path_contributions,
                'loss_contribution': loss_contributions,
                'accumulated_path_integral': accumulated_path_integral,
                'accumulated_importance': accumulated_importance, 
            }
        filename = os.path.join(args.output_dir, f'checkpoint_{epoch+1}.pth.tar')
        torch.save(save_dict,filename)
        print(f"save checkpoint to {filename}")
        accumulated_importance = None
        accumulated_path_integral = None
        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #         'effe_direction' : effe_direction,
        #         'path_contribution': path_contributions,
        #         'loss_contribution': loss_contributions,
        #         'contribution': contributions,
        #         'accumulated_importance': accumulated_importance, 
        #     }, is_best, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, torch.argmax(target, dim=1), topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    filename = os.path.join(args.output_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.output_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()