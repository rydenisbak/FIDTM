from __future__ import division
from config import return_args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = return_args.gpu_id


import warnings

import torch.utils.data
from utils import *
from tqdm import tqdm
import numpy as np
from train_baseline import get_criterion
from py_flower import tools as flower_tools
from test_vl import LMDS_counting, AverageMeter, get_data_loaders


warnings.filterwarnings('ignore')


def get_model(args):

    class Model(object):
        def __init__(self, flower_runtime, flower_model, args):
            self.flower_runtime = flower_runtime
            self.flower_model = flower_model
            self.args = args

        def __call__(self, data: torch.Tensor) -> torch.Tensor:
            data = data.cpu().float().numpy()
            output = self.flower_runtime.run(self.flower_model, {'data': data})
            output = torch.from_numpy(output.outputs[0])
            if int(args['gpu_id']) >= 0:
                output = output.cuda()
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                output = output.to('mps')
            return output

    model = Model(flower_tools.SimpleRuntime(), flower_tools.load_plan(args['flower_plan']), args)
    return model


def main(args):
    model, test_loaders, criterion = get_model(args), get_data_loaders(args), get_criterion(args)

    for root, test_loader in zip(args['val_roots'], test_loaders):
        print('\n' + ('+' * 100))
        print(f'test {root}')
        validate(test_loader, model, criterion, args)
        print('+' * 100)
    print(args['flower_plan'])


@torch.no_grad()
def validate(test_loader, model, criterion, args):
    batch_size, mae_100, visi, index = 1, 0., [], 0
    mae_95_105,  mae_90_110, = 0., 0.

    for i_batch, (fname, img, fidt_map, kpoint) in enumerate(tqdm(test_loader)):

        d6 = model(img)
        if int(args['gpu_id']) >= 0:
            fidt_map = fidt_map.cuda()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            fidt_map = fidt_map.to('mps')

        '''return counting and coordinates'''
        count100, count90_110, count95_105, pred_kpoint = LMDS_counting(d6, args)

        gt_count = torch.sum(kpoint).item()

        local_mae_100 = abs(gt_count - count100)
        mae_100 += local_mae_100

        local_mae_95_105 = abs(gt_count - count95_105)
        mae_95_105 += local_mae_95_105

        local_mae_90_110 = abs(gt_count - count90_110)
        mae_90_110 += local_mae_90_110

    mae_100, mae_95_105, mae_90_110 = [mae / (len(test_loader) * batch_size)
                                       for mae in [mae_100, mae_95_105, mae_90_110]]

    print(' \n* MAE_100 {mae:.3f}'.format(mae=mae_100))
    print(' \n* MAE_95_105 {mae:.3f}'.format(mae=mae_95_105))
    print(' \n* MAE_90_110 {mae:.3f}'.format(mae=mae_90_110))


if __name__ == '__main__':
    params = vars(return_args)
    print(params)
    main(params)
