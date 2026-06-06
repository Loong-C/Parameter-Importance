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
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.utils import * 
from data.sub_dataset import Sub_Dataset, Sub_Dataset_Proportional
import model.mae as mae
import copy
import util.lr_decay as lrd
import util.lr_sched as lr_sched



# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,4,5'

parser = argparse.ArgumentParser(description='PyTorch ImageNet PreTraining use MAE')
parser.add_argument('--data_path', default = '/home/ljl/Datasets/ImageNet/',
                    help = 'path to dataset')
parser.add_argument('--batch_size',default=64, type = int)
parser.add_argument('--sub_batch_size',default = 50, type=int, metavar='N',
                    help='mini-batch size of sub dataloader')
parser.add_argument('--accum_iter', default=1, type=int,
                    help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
parser.add_argument('--train_Proportion', default = 1., type = float)

# Model parameters
parser.add_argument('-a', '--arch', metavar='ARCH', default='mae_vit_small_patch16')
parser.add_argument('--epochs', default = 400, type=int)

parser.add_argument('--input_size', default=224, type=int,
                    help='images input size')

parser.add_argument('--mask_ratio', default=0.75, type=float,
                    help='Masking ratio (percentage of removed patches).')

parser.add_argument('--norm_pix_loss', action='store_true',
                    help='Use (per-patch) normalized pixels as targets for computing loss')
parser.set_defaults(norm_pix_loss=False)

# Optimizer parameters
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')

parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate (absolute lr)')
parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')

parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                    help='epochs to warmup LR')

parser.add_argument('--output_dir', default='./output',type = str,
                    help='path where to save, empty for no saving')
parser.add_argument('--log_dir', default='./output_dir',
                    help='path where to tensorboard log')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--resume', default='',
                    help='resume from checkpoint')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--workers', default=16, type=int)
parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
parser.set_defaults(pin_mem=True)

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


# distributed training parameters
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
    batch_size = args.batch_size
    args.gpu = gpu
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
    model = mae.__dict__[args.arch](
        norm_pix_loss = args.norm_pix_loss
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
    
    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    train_dataset = Sub_Dataset_Proportional(
        full_dataset = full_dataset,
        imagenet_root = os.path.join(args.data_path, 'train'),
        percent_per_class=args.train_Proportion, 
        num_classes=1000,
        transform = transform_train,
    )
    # 用来生成评估loss contribution的batches
    sub_dataset = Sub_Dataset(
        full_dataset = full_dataset,
        imagenet_root = os.path.join(args.data_path, 'train'),
        num_classes = 1000,
        images_per_class = 10,
        transform = transform_train,
        )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        sub_sampler = torch.utils.data.distributed.DistributedSampler(sub_dataset)
    else:
        train_sampler = None
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
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    effe_batch_size = batch_size * args.accum_iter
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * effe_batch_size / 256

    print(f"effe_batch_size:{effe_batch_size},blr:{args.blr}")
    print("actual lr: %.2e" % args.lr)

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    
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
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True


    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        accu_iter = args.accum_iter

        # train for one epoch =====================================
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        end = time.time()
        for data_iter_step, (images, _) in enumerate(train_loader):
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % args.accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)

            ####################################################
            ## loss before
            sub_inputs, _ = batches[args.iter % len(batches)]
            if args.gpu is not None:
                sub_inputs = sub_inputs.cuda(args.gpu, non_blocking=True)
            with torch.no_grad():
                loss_before, _, _ ,ids_shuffle = model(sub_inputs,mask_ratio = args.mask_ratio)
            loss_before = loss_before.item()
            ##########################################################

            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            loss,_,_,ids_train = model(images, mask_ratio = args.mask_ratio)
            losses.update(loss.item(), images.size(0))
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
            
            path_integral = compute_importance_mae(args, model, gradients, delta_param, inputs = images, ids_shuffle = ids_train)
            
            # # 计算 loss_contribution
            if accumulated_path_integral is None:
                accumulated_path_integral = path_integral
            else:
                accumulated_path_integral = add_list_tensor(accumulated_path_integral, path_integral)

            with torch.no_grad():
                loss_after, _, _, _ = model(sub_inputs, mask_ratio = args.mask_ratio, ids_shuffle = ids_shuffle)
            loss_after = loss_after.item()
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
            
            
            # # importance = compute_importance_mae()
            
            # importance = rescale(importance, contribution)
            # importance = checknan(importance)
            importance = rescale_importance_by_contribution(accumulated_path_integral, loss_contribution, path_contribution, tau_1, tau_2)
            importance = checknan(importance)

            # flag = False
            # for i,c in enumerate(importance):
            #     if torch.isnan(c).any():
            #         print(data_iter_step, i, contribution)
            #         flag = True
            # if flag:
            #     exit()

            if accumulated_importance is None:
                accumulated_importance = importance
            else:
                accumulated_importance = add_list_tensor(accumulated_importance, importance)

            args.iter += 1
            # ===================================================================
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if data_iter_step % args.print_freq == 0:
                progress.display(data_iter_step)

        # logging 
        torch.cuda.synchronize()

        # ======================== training status =================================
        save_dict = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'effe_direction' : effe_direction,
                'path_contribution': path_contributions,
                'loss_contribution': loss_contributions,
                'contribution': contributions,
                'accumulated_importance': accumulated_importance, 
                'accumulated_path_integral': accumulated_path_integral,
            }
        filename = os.path.join(args.output_dir, 'checkpoint.pth.tar')
        torch.save(save_dict,filename)
        if epoch %  5 == 0:
            shutil.copyfile(filename, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth.tar'))

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

if __name__ == '__main__':
    main()