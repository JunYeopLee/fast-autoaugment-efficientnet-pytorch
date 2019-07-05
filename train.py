import os
import fire
import time
import json
import random
from pprint import pprint

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from networks import *
from utils import *


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
    transform = get_train_transform(args, model, log_dir)
    val_transform = get_valid_transform(args, model)
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
        _train_res = train_step(args, model, optimizer, scheduler, criterion, batch, step, writer)

        if step % args.print_step == 0:
            print('\n[+] Training step: {}/{}\tTraining epoch: {}/{}\tElapsed time: {:.2f}min\tLearning rate: {}'.format(
                step, args.max_step, current_epoch, max_epoch, (time.time()-start_t)/60, optimizer.param_groups[0]['lr']))
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step=step)
            writer.add_scalar('train/acc1', _train_res[0], global_step=step)
            writer.add_scalar('train/acc5', _train_res[1], global_step=step)
            writer.add_scalar('train/loss', _train_res[2], global_step=step)
            writer.add_scalar('train/forward_time', _train_res[3], global_step=step)
            writer.add_scalar('train/backward_time', _train_res[4], global_step=step)
            print('  Acc@1 : {:.3f}%'.format(_train_res[0].data.cpu().numpy()[0]*100))
            print('  Acc@5 : {:.3f}%'.format(_train_res[1].data.cpu().numpy()[0]*100))
            print('  Loss : {}'.format(_train_res[2].data))
            print('  FW Time : {:.3f}ms'.format(_train_res[3]*1000))
            print('  BW Time : {:.3f}ms'.format(_train_res[4]*1000))

        if step % args.val_step == args.val_step-1:
            valid_loader = iter(get_dataloader(args, valid_dataset))
            _valid_res = validate(args, model, criterion, valid_loader, step, writer)
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


if __name__ == '__main__':
    fire.Fire(train)
