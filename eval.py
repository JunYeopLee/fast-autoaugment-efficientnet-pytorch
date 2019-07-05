import os
import fire
import json
from pprint import pprint

import torch
import torch.nn as nn

from utils import *


def eval(model_path):
    print('\n[+] Parse arguments')
    kwargs_path = os.path.join(model_path, 'kwargs.json')
    kwargs = json.loads(open(kwargs_path).read())
    args, kwargs = parse_args(kwargs)
    pprint(args)
    device = torch.device('cuda' if args.use_cuda else 'cpu')

    print('\n[+] Create network')
    model = select_model(args)
    optimizer = select_optimizer(args, model)
    criterion = nn.CrossEntropyLoss()
    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    print('\n[+] Load model')
    weight_path = os.path.join(model_path, 'model', 'model.pt')
    model.load_state_dict(torch.load(weight_path))

    print('\n[+] Load dataset')
    test_transform = get_valid_transform(args, model)
    test_dataset = get_dataset(args, test_transform, 'test')
    test_loader = iter(get_dataloader(args, test_dataset))

    print('\n[+] Start testing')
    _test_res = validate(args, model, criterion, test_loader, step=0, writer=None)

    print('\n[+] Valid results')
    print('  Acc@1 : {:.3f}%'.format(_test_res[0].data.cpu().numpy()[0]*100))
    print('  Acc@5 : {:.3f}%'.format(_test_res[1].data.cpu().numpy()[0]*100))
    print('  Loss : {:.3f}'.format(_test_res[2].data))
    print('  Infer Time(per image) : {:.3f}ms'.format(_test_res[3]*1000 / len(test_dataset)))


if __name__ == '__main__':
    fire.Fire(eval)
