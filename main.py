import os
import fire
import time
import json
import random
import importlib
import collections
import pickle as cp

import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint

from networks import *

DATASET_PATH = './data/'
current_epoch = 0


def concat_image_features(image, features, max_features=3):
    _, h, w = image.shape

    max_features = min(features.size(0), max_features)
    image_feature = image.clone()

    for i in range(max_features):
        feature = features[i:i+1]
        _min, _max = torch.min(feature), torch.max(feature)
        feature = (feature - _min) / (_max - _min + 1e-6)
        feature = torch.cat([feature]*3, 0)
        feature = feature.view(1, 3, feature.size(1), feature.size(2))
        feature = F.upsample(feature, size=(h,w), mode="bilinear")
        feature = feature.view(3, h, w)
        image_feature = torch.cat((image_feature, feature), 2)

    return image_feature


def get_model_name(args):
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%B_%d_%H:%M:%S")
    model_name = '__'.join([date_time, args.network, str(args.seed)])
    return model_name


def dict_to_namedtuple(d):
    Args = collections.namedtuple('Args', sorted(d.keys()))

    for k,v in d.items():
        if type(v) is dict:
            d[k] = dict_to_namedtuple(v)

        elif type(v) is str:
            try:
                d[k] = eval(v)
            except:
                d[k] = v

    args = Args(**d)
    return args


def parse_args(kwargs):
    # combine with default args
    kwargs['dataset'] =  kwargs['dataset'] if 'dataset' in kwargs else 'cifar10'
    kwargs['network'] =  kwargs['network'] if 'network' in kwargs else 'resnet_cifar10'
    kwargs['optimizer'] = kwargs['optimizer'] if 'optimizer' in kwargs else 'adam'
    kwargs['learning_rate'] = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.1
    kwargs['seed'] =  kwargs['seed'] if 'seed' in kwargs else None
    kwargs['use_cuda'] =  kwargs['use_cuda'] if 'use_cuda' in kwargs else True
    kwargs['use_cuda'] =  kwargs['use_cuda'] and torch.cuda.is_available()
    kwargs['num_workers'] = kwargs['num_workers'] if 'num_workers' in kwargs else 4
    kwargs['print_step'] = kwargs['print_step'] if 'print_step' in kwargs else 2000
    kwargs['val_step'] = kwargs['val_step'] if 'val_step' in kwargs else 2000
    kwargs['scheduler'] = kwargs['scheduler'] if 'scheduler' in kwargs else 'exp'
    kwargs['batch_size'] = kwargs['batch_size'] if 'batch_size' in kwargs else 128
    kwargs['start_step'] = kwargs['start_step'] if 'start_step' in kwargs else 0
    kwargs['max_step'] = kwargs['max_step'] if 'max_step' in kwargs else 64000
    kwargs['fast_auto_augment'] = kwargs['fast_auto_augment'] if 'fast_auto_augment' in kwargs else False

    # to named tuple
    args = dict_to_namedtuple(kwargs)
    return args, kwargs


def select_model(args):
    if args.network in models.__dict__:
        backbone = models.__dict__[args.network]()
        model = BaseNet(backbone, args)
    else:
        Net = getattr(importlib.import_module('networks.{}'.format(args.network)), 'Net')
        model = Net(args)

    print(model)
    return model


def select_optimizer(args, model):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0001)
    elif args.optimizer == 'rms':
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise Exception('Unknown Optimizer')
    return optimizer


def select_scheduler(args, optimizer):
    if not args.scheduler or args.scheduler == 'None':
        return None
    elif args.scheduler =='clr':
        return torch.optim.lr_scheduler.CyclicLR(optimizer, 0.01, 0.015, mode='triangular2', step_size_up=250000, cycle_momentum=False)
    elif args.scheduler =='exp':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999283, last_epoch=-1)
    else:
        raise Exception('Unknown Scheduler')


def get_dataset(args, transform, split='train'):
    assert split in ['train', 'val']

    if args.dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(DATASET_PATH,
                                               train=(split is 'train'),
                                               transform=transform,
                                               download=True)
    elif args.dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(DATASET_PATH,
                                                split=split,
                                                transform=transform,
                                                download=(split is 'val'))
    else:
        raise Exception('Unknown dataset')

    return dataset


def get_dataloader(args, dataset, shuffle=False, pin_memory=True):
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              shuffle=shuffle,
                                              num_workers=args.num_workers,
                                              pin_memory=pin_memory)
    return data_loader


def get_inf_dataloader(args, dataset):
    global current_epoch
    data_loader = iter(get_dataloader(args, dataset, shuffle=True))

    while True:
        try:
            batch = next(data_loader)

        except StopIteration:
            current_epoch += 1
            data_loader = iter(get_dataloader(args, dataset, shuffle=True))
            batch = next(data_loader)

        yield batch


def get_transform(args, model, log_dir=None):
    if args.fast_auto_augment:
        from fast_auto_augment import fast_auto_augment
        transform, val_transform = fast_auto_augment(args, model, K=4, B=100, num_process=4)
        if log_dir:
            cp.dump(transform, open(os.path.join(log_dir, 'augmentation.cp'), 'wb'))

    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        val_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ])

    elif args.dataset == 'imagenet':
        resize_h, resize_w = model.img_size[0], int(model.img_size[1]*1.875)
        transform = transforms.Compose([
            transforms.Resize([resize_h, resize_w]),
            transforms.RandomCrop(model.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        val_transform = transforms.Compose([
            transforms.Resize([resize_h, resize_w]),
            transforms.ToTensor()
        ])

    else:
        raise Exception('Unknown Dataset')

    return transform, val_transform


def train(**kwargs):
    print('\n[+] Parse arguments')
    args, kwargs = parse_args(kwargs)
    pprint(args)
    device = torch.device('cuda' if args.use_cuda else 'cpu')

    print('\n[+] Create log dir')
    model_name = get_model_name(args)
    log_dir = os.path.join('./runs', model_name)
    os.makedirs(os.path.join(log_dir, 'model'))
    json.dump(kwargs, open(os.path.join(log_dir, 'kwargs.json'), 'w'))
    writer = SummaryWriter(log_dir=log_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    print('\n[+] Create network')
    model = select_model(args)
    optimizer = select_optimizer(args, model)
    scheduler = select_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss()
    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    #writer.add_graph(model)

    print('\n[+] Load dataset')
    transform, val_transform = get_transform(args, model, log_dir)
    train_dataset = get_dataset(args, transform, 'train')
    valid_dataset = get_dataset(args, val_transform, 'val')
    train_loader = iter(get_inf_dataloader(args, train_dataset))
    max_epoch = len(train_dataset) // args.batch_size
    best_acc = -1

    print('\n[+] Start training')
    if torch.cuda.device_count() > 1:
        print('\n[+] Use {} GPUs'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    start_t = time.time()
    for step in range(args.start_step, args.max_step):
        batch = next(train_loader)
        _train_res = _train(args, model, optimizer, scheduler, criterion, batch, step, writer)

        if step % args.print_step == 0:
            print('\n[+] Training step: {}/{}\tTraining epoch: {}/{}\tElapsed time: {:.2f}min\tLearning rate: {}'.format(
                step, args.max_step, current_epoch, max_epoch, (time.time()-start_t)/60, optimizer.param_groups[0]['lr']))
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step=step)
            writer.add_scalar('train/acc1', _train_res[0], global_step=step)
            writer.add_scalar('train/acc5', _train_res[1], global_step=step)
            writer.add_scalar('train/loss', _train_res[2], global_step=step)
            print('  Acc@1 : {:.3f}%'.format(_train_res[0].data.cpu().numpy()[0]*100))
            print('  Acc@5 : {:.3f}%'.format(_train_res[1].data.cpu().numpy()[0]*100))
            print('  Loss : {}'.format(_train_res[2].data))

        if step % args.val_step == args.val_step-1:
            valid_loader = iter(get_dataloader(args, valid_dataset))
            _valid_res = _validate(args, model, criterion, valid_loader, step, writer)
            print('\n[+] Valid results')
            writer.add_scalar('valid/acc1', _valid_res[0], global_step=step)
            writer.add_scalar('valid/acc5', _valid_res[1], global_step=step)
            writer.add_scalar('valid/loss', _valid_res[2], global_step=step)
            print('  Acc@1 : {:.3f}%'.format(_valid_res[0].data.cpu().numpy()[0]*100))
            print('  Acc@5 : {:.3f}%'.format(_valid_res[1].data.cpu().numpy()[0]*100))
            print('  Loss : {}'.format(_valid_res[2].data))

            if _valid_res[0] > best_acc:
                best_acc = _valid_res[0]
                torch.save(model.state_dict(), os.path.join(log_dir, "model","model.pt"))
                print('\n[+] Model saved')

    writer.close()


def _train(args, model, optimizer, scheduler, criterion, batch, step, writer, device=None):
    model.train()
    images, target = batch

    if device:
        images = images.to(device)
        target = target.to(device)

    elif args.use_cuda:
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

    # compute output
    output, first = model(images)
    loss = criterion(output, target)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    acc1 /= images.size(0)
    acc5 /= images.size(0)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler: scheduler.step()

    if writer and step % args.print_step == 0:
        for j in range(10):
            writer.add_image('train/input_image', concat_image_features(images[j], first[j]), global_step=step)

    return acc1, acc5, loss


def _validate(args, model, criterion, valid_loader, step, writer, device=None):
    # switch to evaluate mode
    model.eval()

    acc1, acc5 = 0, 0
    samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(valid_loader):

            if device:
                images = images.to(device)
                target = target.to(device)

            elif args.use_cuda is not None:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output, first = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            _acc1, _acc5 = accuracy(output, target, topk=(1, 5))
            acc1 += _acc1
            acc5 += _acc5
            samples += images.size(0)

    acc1 /= samples
    acc5 /= samples

    if writer:
        writer.add_image('valid/input_image', concat_image_features(images[0], first[0]), global_step=step)
        writer.add_image('valid/input_image', concat_image_features(images[1], first[1]), global_step=step)
        writer.add_image('valid/input_image', concat_image_features(images[2], first[2]), global_step=step)

    return acc1, acc5, loss


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


if __name__ == '__main__':
    fire.Fire(train)
