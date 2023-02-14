import os
import json
import pprint as pp
import random
import numpy as np
from datetime import date

import torch
import torch.backends.cudnn as cudnn

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def setup_train(args):
    set_up_gpu(args)

    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)

    pp.pprint({k: v for k, v in vars(args).items() if v is not None}, width=1)
    return export_root

def export_experiments_config_as_json(args, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=2)

def set_up_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
    args.num_gpu = len(args.device_idx.split(","))

def create_experiment_export_folder(args):
    experiment_dir, experiment_description = args.experiment_dir, args.experiment_description
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)
    os.mkdir(experiment_path)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path

def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path

def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx

class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class PrintInputShape:
    def __init__(self, limit=1):
        self.limit = limit
        self.cnt = 1

    # @classmethod
    def print(self, t:np.array, notation):
        if self.cnt <= self.limit:
            print(f"{notation} : {len(t.shape)}d", end='\n')
            if len(t.shape) == 1:
                self.print_1d(t)
                self.print_0d_shape(len(t))
            elif len(t.shape) == 2:
                self.print_2d(t)
            elif len(t.shape) == 3:
                self.print_3d(t)
            else:
                print(f"1d array or over 3d")
            print()
        self.cnt += 1

    def print_0d_shape(self, dim):
        print(f"\t<-- {dim} -->")

    def print_1d(self, data_1d, ws='', arrow=''):
        sample = f"[{data_1d[0]:5}, {data_1d[1]:5}, ..., {data_1d[-2]:5}, {data_1d[-1]:5}]"
        print(f"{ws}{arrow}\t{sample}")
        

    def print_2d(self, t, ws=''):
        t_trim_a = t.clone().detach()[:3]
        t_trim_b = t.clone().detach()[-3:]
        t_trim = np.concatenate([t_trim_a, t_trim_b], axis=0)
        for line, data_1d in enumerate(t_trim, start=1):
            if line == 1:
                self.print_1d(data_1d, ws, arrow='^')
            elif line == 2:
                self.print_1d(data_1d, ws, arrow='|')
            elif line == (len(t_trim)-1):
                self.print_1d(data_1d, ws, arrow='|')
            elif line == len(t_trim):
                self.print_1d(data_1d, ws, arrow='v')
            else:
                self.print_1d(data_1d, ws, arrow='|')

            if line == len(t_trim)//2:  # 중간에 숫자 끼워넣기
                print(f"{ws}{t.size()[0]}\t\t\t...")
        self.print_0d_shape(t.size()[1])

    def print_3d(self, t):
        print(f"Input's shape : {t.size()}")
        ws = ''
        print(f"{ws}\\")
        ws += ' '
        print(f"{ws}{t.size()[0]}")
        ws += ' '
        print(f"{ws}\\")

        print_2d_shape(self, t[-1], ws=ws)

