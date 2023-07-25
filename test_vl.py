from __future__ import division
import os
from config import return_args, args
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


import warnings

import torch.utils.data
from Networks.HR_Net.seg_hrnet import get_seg_model
import math
from image import *
from utils import *
from tqdm import tqdm
import numpy as np
from data.nwpu_dataset import NWPUCrowdDataset
from mmcv.cnn.utils import get_model_complexity_info
import segmentation_models_pytorch as smp
from Networks.vl_seg_models.models import VLSegModel
from train_baseline import get_criterion


warnings.filterwarnings('ignore')


def get_data_loaders(args):
    test_datasets = [NWPUCrowdDataset(root, 'val', args) for root in args['val_roots']]
    common_args = dict(
        drop_last=False,
        pin_memory=False,
        num_workers=args['workers'],
        persistent_workers=False,
        batch_size=1,
        shuffle=False,
    )

    return [torch.utils.data.DataLoader(item, **common_args) for item in test_datasets]


def get_model(args):
    if args['model_type'].lower() == 'hrnet':
        model = get_seg_model(train=True, norm_eval=args['norm_eval'])
    elif args['model_type'] == 'vl_hrnet':
        model = get_seg_model(train=False, model_name='flower')
    elif args['model_type'].startswith('vl_'):
        model = VLSegModel(args)
    else:
        model = smp.PAN(
            encoder_name=args['model_type'],
            encoder_weights=None,
            classes=1
        )

    model.eval()

    print("=> loading checkpoint '{}'".format(args['load_checkpoint']))
    sd = torch.load(args['load_checkpoint'], 'cpu')
    if 'state_dict' in sd:
        sd = sd['state_dict']
        sd = {key.replace('module.', ''): val for key, val in sd.items()}
    model.load_state_dict(sd, strict=True)

    if args['to_avx2']:
        model.to_avx2()
    if args['onnx_file'] is not None:
        input_shape = 1, 3, 640, 640
        feat = torch.randn(input_shape)
        torch.onnx.export(model, feat, args['onnx_file'], verbose=True, opset_version=9)
    flops, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False)
    print(f'{args["model_type"]} flops: {flops}, params: {params}')

    if int(args['gpu_id']) >= 0:
        model = model.cuda()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        model = model.to('mps')

    return model


def main(args):
    model, test_loaders, criterion = get_model(args), get_data_loaders(args), get_criterion(args)

    fp16_args = {
        'scaler': torch.cuda.amp.GradScaler(init_scale=2.**10, enabled=True) if args['fp16'] else None,
        'device_type': 'cuda' if int(args['gpu_id']) >= 0 else 'cpu',
        'dtype': torch.float16 if args['fp16'] else torch.float32,
    }

    for root, test_loader in zip(args['val_roots'], test_loaders):
        print('\n' + ('+' * 100))
        print(f'test {root}')
        validate(test_loader, model, criterion, args, fp16_args)
        print('+' * 100)
    print(args['load_checkpoint'])


@torch.no_grad()
def validate(test_loader, model, criterion, args, fp16_args):
    model.eval()
    batch_size, mae_100, visi, index = 1, 0., [], 0
    mae_95_105,  mae_90_110, = 0., 0.
    total_losses, ssim_losses, mse_losses, norm_ssim_losses = [AverageMeter() for _ in range(4)]

    for i_batch, (fname, img, fidt_map, kpoint) in enumerate(tqdm(test_loader)):

        if args['fp16']:
            with torch.autocast(device_type=fp16_args['device_type'], dtype=fp16_args['dtype'], enabled=True):
                d6 = model(img.half().cuda())
                mse_loss, ssim_loss = criterion(d6, fidt_map.half().cuda(), kpoint)
        else:
            if int(args['gpu_id']) >= 0:
                img = img.cuda()
                fidt_map = fidt_map.cuda()
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                img = img.to('mps')
                fidt_map = fidt_map.to('mps')

            d6 = model(img)
            mse_loss, ssim_loss = criterion(d6, fidt_map, kpoint)

        if args['loss_type'] == 'MSE':
            loss = mse_loss
        else:
            loss = mse_loss + ssim_loss

        total_losses.update(loss.item())
        mse_losses.update(mse_loss.item())
        ssim_losses.update(ssim_loss.item())
        norm_ssim_losses.update(ssim_loss.item() / criterion.ssim_coefficient)

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
    print(f'val_loss/total_loss {total_losses.avg}')
    print(f'val_loss/mse_loss {mse_losses.avg}')
    print(f'val_loss/ssim_loss {ssim_losses.avg}')
    print(f'val_loss/norm_ssim_losses {norm_ssim_losses.avg}')


def LMDS_counting(input, args):
    input_max = torch.max(input).item()

    ''' find local maxima'''
    if args['dataset'] == 'UCF_QNRF' :
        input = nn.functional.avg_pool2d(input, (3, 3), stride=1, padding=1)
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    else:
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    '''set the pixel valur of local maxima as 1 for counting'''

    input_clone = input.clone()
    counts = []
    thrs_range = [90, 111]
    for thr in range(*thrs_range):
        if thr > thrs_range[0]:
            input = input_clone.clone()
        input[input < thr / 255.0 * input_max] = 0
        input[input > 0] = 1
        ''' negative sample'''
        if input_max < (thr / 1000):
            input = input * 0
        counts.append(int(torch.sum(input).item()))
        if thr == 100:
            kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    count100, count90_110, count95_105 = counts[10], np.mean(counts), np.mean(counts[5:16])

    return count100, count90_110, count95_105, kpoint


def generate_point_map(kpoint, f_loc, rate=1):
    '''obtain the location coordinates'''
    pred_coor = np.nonzero(kpoint)

    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)

    for data in coord_list:
        f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
    f_loc.write('\n')

    return point_map


def generate_bounding_boxes(kpoint, fname):
    '''change the data path'''
    Img_data = cv2.imread(
        '/home/dkliang/projects/synchronous/datasets/ShanghaiTech/part_A_final/test_data/images/' + fname[0])
    ori_Img_data = Img_data.copy()

    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    distances, locations = tree.query(pts, k=4)
    for index, pt in enumerate(pts):
        pt2d = np.zeros(kpoint.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if np.sum(kpoint) > 1:
            sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
        else:
            sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
        sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.05)

        if sigma < 6:
            t = 2
        else:
            t = 2
        Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
                                 (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)

    return ori_Img_data, Img_data


def show_map(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


if __name__ == '__main__':
    params = vars(return_args)
    print(params)
    main(params)
