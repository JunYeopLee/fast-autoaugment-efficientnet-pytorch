import copy
import json
import time
import torch
import random
import torchvision.transforms as transforms

from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
from concurrent.futures import ProcessPoolExecutor

from transforms import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from utils import *


DEFALUT_CANDIDATES = [
    ShearXY,
    TranslateXY,
    Rotate,
    AutoContrast,
    Invert,
    Equalize,
    Solarize,
    Posterize,
    Contrast,
    Color,
    Brightness,
    Sharpness,
    Cutout,
#     SamplePairing,
]


def train_child(args, model, dataset, subset_indx, device=None):
    optimizer = select_optimizer(args, model)
    scheduler = select_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss()

    dataset.transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()])
    subset = Subset(dataset, subset_indx)
    data_loader = get_inf_dataloader(args, subset)

    if device:
        model = model.to(device)
        criterion = criterion.to(device)

    elif args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

        if torch.cuda.device_count() > 1:
            print('\n[+] Use {} GPUs'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    start_t = time.time()
    for step in range(args.start_step, args.max_step):
        batch = next(data_loader)
        _train_res = train_step(args, model, optimizer, scheduler, criterion, batch, step, None, device)

        if step % args.print_step == 0:
            print('\n[+] Training step: {}/{}\tElapsed time: {:.2f}min\tLearning rate: {}\tDevice: {}'.format(
                step, args.max_step,(time.time()-start_t)/60, optimizer.param_groups[0]['lr'], device))

            print('  Acc@1 : {:.3f}%'.format(_train_res[0].data.cpu().numpy()[0]*100))
            print('  Acc@5 : {:.3f}%'.format(_train_res[1].data.cpu().numpy()[0]*100))
            print('  Loss : {}'.format(_train_res[2].data))

    return _train_res


def validate_child(args, model, dataset, subset_indx, transform, device=None):
    criterion = nn.CrossEntropyLoss()

    if device:
        model = model.to(device)
        criterion = criterion.to(device)

    elif args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    dataset.transform = transform
    subset = Subset(dataset, subset_indx)
    data_loader = get_dataloader(args, subset, pin_memory=False)

    return validate(args, model, criterion, data_loader, 0, None, device)


def get_next_subpolicy(transform_candidates, op_per_subpolicy=2):
    n_candidates = len(transform_candidates)
    subpolicy = []

    for i in range(op_per_subpolicy):
        indx = random.randrange(n_candidates)
        prob = random.random()
        mag = random.random()
        subpolicy.append(transform_candidates[indx](prob, mag))

    subpolicy = transforms.Compose([
        *subpolicy,
        transforms.Resize(32),
        transforms.ToTensor()])

    return subpolicy


def search_subpolicies(args, transform_candidates, child_model, dataset, Da_indx, B, device):
    subpolicies = []

    for b in range(B):
        subpolicy = get_next_subpolicy(transform_candidates)
        val_res = validate_child(args, child_model, dataset, Da_indx, subpolicy, device)
        subpolicies.append((subpolicy, val_res[2]))

    return subpolicies


def search_subpolicies_hyperopt(args, transform_candidates, child_model, dataset, Da_indx, B, device):

    def _objective(sampled):
        subpolicy = [transform(prob, mag)
                     for transform, prob, mag in sampled]

        subpolicy = transforms.Compose([
            transforms.Resize(32),
            *subpolicy,
            transforms.ToTensor()])

        val_res = validate_child(args, child_model, dataset, Da_indx, subpolicy, device)
        loss = val_res[2].cpu().numpy()
        return {'loss': loss, 'status': STATUS_OK }

    space = [(hp.choice('transform1', transform_candidates), hp.uniform('prob1', 0, 1.0), hp.uniform('mag1', 0, 1.0)),
             (hp.choice('transform2', transform_candidates), hp.uniform('prob2', 0, 1.0), hp.uniform('mag2', 0, 1.0))]

    trials = Trials()
    best = fmin(_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=B,
                trials=trials)

    subpolicies = []
    for t in trials.trials:
        vals = t['misc']['vals']
        subpolicy = [transform_candidates[vals['transform1'][0]](vals['prob1'][0], vals['mag1'][0]),
                     transform_candidates[vals['transform2'][0]](vals['prob2'][0], vals['mag2'][0])]
        subpolicy = transforms.Compose([
            ## baseline augmentation
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            ## policy
            *subpolicy,
            ## to tensor
            transforms.ToTensor()])
        subpolicies.append((subpolicy, t['result']['loss']))

    return subpolicies


def get_topn_subpolicies(subpolicies, N=10):
    return sorted(subpolicies, key=lambda subpolicy: subpolicy[1])[:N]


def process_fn(args_str, model, dataset, Dm_indx, Da_indx, T, transform_candidates, B, N, k):
    kwargs = json.loads(args_str)
    args, kwargs = parse_args(kwargs)
    device_id = k % torch.cuda.device_count()
    device = torch.device('cuda:%d' % device_id)
    _transform = []

    print('[+] Child %d training strated (GPU: %d)' % (k, device_id))

    # train child model
    child_model = copy.deepcopy(model)
    train_res = train_child(args, child_model, dataset, Dm_indx, device)

    # search sub policy
    for t in range(T):
        #subpolicies = search_subpolicies(args, transform_candidates, child_model, dataset, Da_indx, B, device)
        subpolicies = search_subpolicies_hyperopt(args, transform_candidates, child_model, dataset, Da_indx, B, device)
        subpolicies = get_topn_subpolicies(subpolicies, N)
        _transform.extend([subpolicy[0] for subpolicy in subpolicies])

    return _transform


def fast_auto_augment(args, model, transform_candidates=None, K=5, B=100, T=2, N=10, num_process=5):
    args_str = json.dumps(args._asdict())
    dataset = get_dataset(args, None, 'trainval')
    num_process = min(torch.cuda.device_count(), num_process)
    transform, futures = [], []

    torch.multiprocessing.set_start_method('spawn', force=True)

    if not transform_candidates:
        transform_candidates = DEFALUT_CANDIDATES

    # split
    Dm_indexes, Da_indexes = split_dataset(args, dataset, K)

    with ProcessPoolExecutor(max_workers=num_process) as executor:
        for k, (Dm_indx, Da_indx) in enumerate(zip(Dm_indexes, Da_indexes)):
            future = executor.submit(process_fn,
                    args_str, model, dataset, Dm_indx, Da_indx, T, transform_candidates, B, N, k)
            futures.append(future)

        for future in futures:
            transform.extend(future.result())

    transform = transforms.RandomChoice(transform)

    return transform
