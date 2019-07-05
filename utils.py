import os
import time
import importlib
import collections
import pickle as cp

import torch
import torchvision
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Subset

from sklearn.model_selection import StratifiedShuffleSplit


DATASET_PATH = './data/'
current_epoch = 0


def split_dataset(args, dataset, k):
    # load dataset
    X = list(range(len(dataset)))
    Y = dataset.targets

    # split to k-fold
    assert len(X) == len(Y)

    def _it_to_list(_it):
        return list(zip(*list(_it)))

    sss = StratifiedShuffleSplit(n_splits=k, random_state=args.seed, test_size=0.1)
    Dm_indexes, Da_indexes = _it_to_list(sss.split(X, Y))

    return Dm_indexes, Da_indexes


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
    kwargs['augment_path'] = kwargs['augment_path'] if 'augment_path' in kwargs else None

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
    assert split in ['train', 'val', 'test', 'trainval']

    if args.dataset == 'cifar10':
        train = split in ['train', 'val', 'trainval']
        dataset = torchvision.datasets.CIFAR10(DATASET_PATH,
                                               train=train,
                                               transform=transform,
                                               download=True)

        if split in ['train', 'val']:
            split_path = os.path.join(DATASET_PATH,
                    'cifar-10-batches-py', 'train_val_index.cp')

            if not os.path.exists(split_path):
                [train_index], [val_index] = split_dataset(args, dataset, k=1)
                split_index = {'train':train_index, 'val':val_index}
                cp.dump(split_index, open(split_path, 'wb'))

            split_index = cp.load(open(split_path, 'rb'))
            dataset = Subset(dataset, split_index[split])

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


def get_train_transform(args, model, log_dir=None):
    if args.fast_auto_augment:
        assert args.dataset == 'cifar10' # TODO: FastAutoAugment for Imagenet

        from fast_auto_augment import fast_auto_augment
        if args.augment_path:
            transform = cp.load(open(args.augment_path, 'rb'))
            os.system('cp {} {}'.format(
                args.augment_path, os.path.join(log_dir, 'augmentation.cp')))
        else:
            transform = fast_auto_augment(args, model, K=4, B=1, num_process=4)
            if log_dir:
                cp.dump(transform, open(os.path.join(log_dir, 'augmentation.cp'), 'wb'))

    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
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

    else:
        raise Exception('Unknown Dataset')

    print(transform)

    return transform


def get_valid_transform(args, model):
    if args.dataset == 'cifar10':
        val_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ])

    elif args.dataset == 'imagenet':
        resize_h, resize_w = model.img_size[0], int(model.img_size[1]*1.875)
        val_transform = transforms.Compose([
            transforms.Resize([resize_h, resize_w]),
            transforms.ToTensor()
        ])

    else:
        raise Exception('Unknown Dataset')

    return val_transform


def train_step(args, model, optimizer, scheduler, criterion, batch, step, writer, device=None):
    model.train()
    images, target = batch

    if device:
        images = images.to(device)
        target = target.to(device)

    elif args.use_cuda:
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

    # compute output
    start_t = time.time()
    output, first = model(images)
    forward_t = time.time() - start_t
    loss = criterion(output, target)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    acc1 /= images.size(0)
    acc5 /= images.size(0)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    start_t = time.time()
    loss.backward()
    backward_t = time.time() - start_t
    optimizer.step()
    if scheduler: scheduler.step()

    if writer and step % args.print_step == 0:
        n_imgs = min(images.size(0), 10)
        for j in range(n_imgs):
            writer.add_image('train/input_image',
                    concat_image_features(images[j], first[j]), global_step=step)

    return acc1, acc5, loss, forward_t, backward_t


def validate(args, model, criterion, valid_loader, step, writer, device=None):
    # switch to evaluate mode
    model.eval()

    acc1, acc5 = 0, 0
    samples = 0
    infer_t = 0

    with torch.no_grad():
        for i, (images, target) in enumerate(valid_loader):

            start_t = time.time()
            if device:
                images = images.to(device)
                target = target.to(device)

            elif args.use_cuda is not None:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output, first = model(images)
            loss = criterion(output, target)
            infer_t += time.time() - start_t

            # measure accuracy and record loss
            _acc1, _acc5 = accuracy(output, target, topk=(1, 5))
            acc1 += _acc1
            acc5 += _acc5
            samples += images.size(0)

    acc1 /= samples
    acc5 /= samples

    if writer:
        n_imgs = min(images.size(0), 10)
        for j in range(n_imgs):
            writer.add_image('valid/input_image',
                    concat_image_features(images[j], first[j]), global_step=step)

    return acc1, acc5, loss, infer_t


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

