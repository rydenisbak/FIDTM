from __future__ import division
import os
from config import return_args, args
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


import warnings

import torch.utils.data

from Networks.HR_Net.seg_hrnet import get_seg_model

from torchvision import transforms
import dataset
import math
from image import *
from utils import *
from os import path as osp

import logging

import time
from tqdm import tqdm
import numpy as np
from losses import MSE_ISSIM
import optim_utils
from tensorboardX import SummaryWriter
from data.shanghai_a import ShanghaiADataset
from data.nwpu_dataset import NWPUCrowdDataset, ConcatDataset
from mmcv.cnn.utils import get_model_complexity_info
import segmentation_models_pytorch as smp
from Networks.vl_seg_models.models import VLSegModel


warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')


def get_criterion(args):
    if args['loss_type'] in ['MSE', 'MSE_SSIM']:
        criterion = MSE_ISSIM(normalize_mse=args['normalize_mse'],
                              ssim_coefficient=args['ssim_coefficient'])
    else:
        raise AttributeError(f'\'{args["loss_type"]}\' is unknown loss type')

    if int(args['gpu_id']) >= 0:
        criterion = criterion.cuda()
        if args['fp16']:
            criterion = criterion.half()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        criterion = criterion.to('mps')

    return criterion


def get_data_loaders_backup(args):
    if args['dataset'] == 'ShanghaiA':
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/qnrf_train.npy'
        test_file = './npydata/qnrf_test.npy'
    elif args['dataset'] == 'JHU':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'NWPU':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        test_list = np.load(outfile).tolist()

    if args['preload_data']:
        train_data = pre_data(train_list, args, train=True)
        test_data = pre_data(test_list, args, train=False)
    else:
        train_data = train_list
        test_data = test_list

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            args=args),
        batch_size=args['batch_size'],
        drop_last=False,
        pin_memory=True,
        num_workers=args['workers'],
        persistent_workers=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(test_data, args['save_path'],
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1,
        drop_last=False,
        num_workers=args['workers'],
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
    )

    return train_loader, test_loader


def get_data_loaders(args):
    if args['dataset'] == 'ShanghaiA':
        train_dataset = ShanghaiADataset(data_file='./npydata/ShanghaiA_train.npy',
                                         mode='train',
                                         args=args)
        val_dataset = ShanghaiADataset(data_file='./npydata/ShanghaiA_test.npy',
                                       mode='val',
                                       args=args)
    elif args['dataset'] == 'NWPU':
        train_dataset = NWPUCrowdDataset(args['nwpu_root'], 'train', args)
        val_dataset = NWPUCrowdDataset(args['nwpu_root'], 'val', args)
    elif args['dataset'] == 'concat':
        train_datasets = [NWPUCrowdDataset(root, 'train', args) for root in args['train_roots']]
        train_dataset = ConcatDataset(train_datasets, args['train_roots'], args['train_weights'])

        val_dataset = NWPUCrowdDataset(args['nwpu_root'], 'val', args)

    common_loader_args = dict(
        drop_last=False,
        pin_memory=True,
        num_workers=args['workers'],
        persistent_workers=True,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        **common_loader_args
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        **common_loader_args
    )

    return train_loader, val_loader


def get_model_optimizer_and_sheduler(args):
    if args['model_type'].lower() == 'hrnet':
        model = get_seg_model(train=True, norm_eval=args['norm_eval'])
    elif args['model_type'].startswith('vl_'):
        model = VLSegModel(args)
    else:
        model = smp.PAN(
            encoder_name=args['model_type'],
            encoder_weights='imagenet',
            classes=1
        )

    model.eval()
    flops, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False)
    print(f'{args["model_type"]} flops: {flops}, params: {params}')

    if args['optimizer'] == 'sgd':
        print('Using sgd optimizer')
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], momentum=.9)
    elif args['optimizer'] == 'adam':
        print('Using adam optimizer')
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    else:
        raise Exception(f'Unknown optimizer type {args["optimizer"]}')

    # set LR scheduler
    if args['lr_scheduler'] == 'multistep':
        lr_steps = [int(x) for x in args['lr_steps'].strip().split(',')]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=0.1)
    elif args['lr_scheduler'] == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    elif args['lr_scheduler'] == 'warmup_cosine':
        lr_scheduler = optim_utils.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args['warmup_epochs'],
                                                                 max_epochs=args['epochs'])
    else:
        raise Exception(f'Unknown LR scheduler {args.lr_scheduler}')

    assert not (args['pre'] and args['load_checkpoint']), 'choose --pre or --load_checkpoint not simultaneously'
    if args['load_checkpoint'] is not None:
        print("=> loading checkpoint '{}'".format(args['load_checkpoint']))
        sd = torch.load(args['load_checkpoint'], 'cpu')['state_dict']
        model.load_state_dict(sd, strict=True)
    if args['pre']:
        print(f'resume from {args["pre"]}')
        if osp.isfile(args['pre']):
            print("=> resume checkpoint '{}'".format(args['pre']))
            # if int(args['gpu_id']) >= 0:
            #     checkpoint = torch.load(args['pre'], 'cuda')
            # else:
            #     checkpoint = torch.load(args['pre'], 'cpu')
            checkpoint = torch.load(args['pre'], 'cpu')
            sd = {key.replace('module.', ''): val for key, val in checkpoint['state_dict'].items()}
            model.load_state_dict(sd, strict=True)
            if not args['only_validate']:
                args['start_epoch'] = checkpoint['epoch']
                args['best_pred'] = checkpoint['best_prec1']
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                # optimizer.load_state_dict(checkpoint['optimizer'])

        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    if int(args['gpu_id']) >= 0:
        model = model.cuda()

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        model = model.to('mps')

    return model, optimizer, lr_scheduler


def main(args):

    model, optimizer, lr_scheduler = get_model_optimizer_and_sheduler(args)

    if args['transforms_type'] == 'ordinary':
        train_loader, test_loader = get_data_loaders_backup(args)
    else:
        train_loader, test_loader = get_data_loaders(args)

    criterion = get_criterion(args)

    fp16_args = {
        'scaler': torch.cuda.amp.GradScaler(init_scale=2.**10, enabled=True) if args['fp16'] else None,
        'device_type': 'cuda' if int(args['gpu_id']) >= 0 else 'cpu',
        'dtype': torch.float16 if args['fp16'] else torch.float32,
    }

    os.makedirs(args['save_path'], exist_ok=True)
    tb_writer = SummaryWriter(log_dir=args['save_path'])
    print(f'best_mae: {args["best_pred"]:.1f}, start_epoch: {args["start_epoch"]}')

    best_epoch = -1
    for epoch in range(args['start_epoch'], args['epochs']):
        if args['only_validate']:
            prec1, visi = validate(test_loader, model, criterion, epoch, args, tb_writer, fp16_args)
            args['best_pred'] = prec1
            print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']), args['save_path'])
            return

        start = time.time()
        train(train_loader, model, criterion, optimizer, epoch, args, tb_writer, fp16_args)
        lr_scheduler.step()
        end1 = time.time()

        '''inference '''
        if (epoch % 1 == 0 or epoch == args['epochs'] - 1) and epoch > args['epochs_wo_logs']:
            prec1, visi = validate(test_loader, model, criterion, epoch, args, tb_writer, fp16_args)

            end2 = time.time()

            is_best = prec1 < args['best_pred']
            args['best_pred'] = min(prec1, args['best_pred'])

            print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']), args['save_path'], end1 - start, end2 - end1)
            if is_best:
                best_epoch = epoch + 1
            print(f' * epoch of best MAE is {best_epoch}')

            if epoch > args['warmup_epochs']:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args['pre'],
                    'state_dict': model.state_dict(),
                    'best_prec1': args['best_pred'],
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                }, visi, is_best, args['save_path'])


def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        # print(fname)
        img, fidt_map, kpoint = load_data_fidt(Img_path, args, train)

        if min(fidt_map.shape[0], fidt_map.shape[1]) < 256 and train:
            # ignore some small resolution images
            continue
        # print(img.size, fidt_map.shape)
        blob = {}
        blob['img'] = img
        blob['kpoint'] = np.array(kpoint)
        blob['fidt_map'] = fidt_map
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

    return data_keys


def train(train_loader, model, criterion, optimizer, epoch, args, tb_writer, fp16_args):
    total_losses, ssim_losses, mse_losses, batch_time, data_time = [AverageMeter() for _ in range(5)]
    norm_ssim_losses = AverageMeter()

    max_lr = max([param['lr'] for param in optimizer.param_groups])
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), max_lr))
    tb_writer.add_scalar('lr', max_lr, epoch)

    model.train()
    if args['norm_eval'] and args['model_type'].lower() != 'hrnet':
        if args['norm_eval_encoder']:
            model.encoder = freeze_bn(model.encoder)
        else:
            model = freeze_bn(model)

    end = time.time()
    for i_batch, (fname, img, fidt_map, kpoint) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end)

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

        if args['fp16']:
            fp16_args['scaler'].scale(loss).backward()
            fp16_args['scaler'].step(optimizer)
            fp16_args['scaler'].update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        total_losses.update(loss.item())
        mse_losses.update(mse_loss.item())
        ssim_losses.update(ssim_loss.item())
        norm_ssim_losses.update(ssim_loss.item() / criterion.ssim_coefficient)

        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{0}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, batch_time=batch_time,
                                                          data_time=data_time, loss=total_losses))
    tb_writer.add_scalar('train/total_loss', total_losses.avg, epoch)
    tb_writer.add_scalar('train/mse_loss', mse_losses.avg, epoch)
    tb_writer.add_scalar('train/ssim_loss', ssim_losses.avg, epoch)
    tb_writer.add_scalar('train/norm_ssim_loss', norm_ssim_losses.avg, epoch)


@torch.no_grad()
def validate(test_loader, model, criterion, epoch, args, tb_writer, fp16_args):
    print('begin test')
    model.eval()

    batch_size, mae_100, mse_100, visi, index = 1, 0., 0., [], 0
    mae_95_105, mse_95_105, mae_90_110, mse_90_110 = 0., 0., 0., 0.
    total_losses, ssim_losses, mse_losses, norm_ssim_losses = [AverageMeter() for _ in range(4)]
    os.makedirs('./local_eval/loc_file', exist_ok=True)

    '''output coordinates'''
    f_loc = open("./local_eval/A_localization.txt", "w+")

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
        count100, count90_110, count95_105, pred_kpoint, f_loc = LMDS_counting(d6, i_batch + 1, f_loc, args)
        point_map = generate_point_map(pred_kpoint, f_loc, rate=1)

        if args['visual']:
            if not os.path.exists(args['save_path'] + '_box/'):
                os.makedirs(args['save_path'] + '_box/')
            ori_img, box_img = generate_bounding_boxes(pred_kpoint, fname)
            show_fidt = show_map(d6.data.cpu().numpy())
            gt_show = show_map(fidt_map.data.cpu().numpy())
            res = np.hstack((ori_img, gt_show, show_fidt, point_map, box_img))
            cv2.imwrite(args['save_path'] + '_box/' + fname[0], res)

        gt_count = torch.sum(kpoint).item()

        local_mae_100 = abs(gt_count - count100)
        mae_100 += local_mae_100
        mse_100 += local_mae_100 ** 2

        local_mae_95_105 = abs(gt_count - count95_105)
        mae_95_105 += local_mae_95_105
        mse_95_105 += local_mae_95_105 ** 2

        local_mae_90_110 = abs(gt_count - count90_110)
        mae_90_110 += local_mae_90_110
        mse_90_110 += local_mae_90_110 ** 2

        # if i % 15 == 0:
        #     print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))
        #     visi.append(
        #         [img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(),
        #          fname])
        #     index += 1

    mae_100, mae_95_105, mae_90_110 = [mae / (len(test_loader) * batch_size)
                                       for mae in [mae_100, mae_95_105, mae_90_110]]
    mse_100, mse_95_105, mse_90_110 = [math.sqrt(mse / (len(test_loader) * batch_size))
                                       for mse in [mse_100, mse_95_105, mse_90_110]]

    print(' \n* MAE_100 {mae:.3f}'.format(mae=mae_100), ' * MSE_100 {mse:.3f}'.format(mse=mse_100))
    print(' \n* MAE_95_105 {mae:.3f}'.format(mae=mae_95_105), ' * MSE_95_105 {mse:.3f}'.format(mse=mse_95_105))
    print(' \n* MAE_90_110 {mae:.3f}'.format(mae=mae_90_110), ' * MSE_90_110 {mse:.3f}'.format(mse=mse_90_110))

    if args['only_validate']:
        print(f'val_loss/total_loss {total_losses.avg}')
        print(f'val_loss/mse_loss {mse_losses.avg}')
        print(f'val_loss/ssim_loss {ssim_losses.avg}')
        print(f'val_loss/norm_ssim_losses {norm_ssim_losses.avg}')
    else:
        tb_writer.add_scalar('val_loss/total_loss', total_losses.avg, epoch)
        tb_writer.add_scalar('val_loss/mse_loss', mse_losses.avg, epoch)
        tb_writer.add_scalar('val_loss/ssim_loss', ssim_losses.avg, epoch)
        tb_writer.add_scalar('val_loss/norm_ssim_losses', norm_ssim_losses.avg, epoch)

        tb_writer.add_scalar('val_count/mae_100', mae_100, epoch)
        tb_writer.add_scalar('val_count/mae_95_105', mae_95_105, epoch)
        tb_writer.add_scalar('val_count/mae_90_110', mae_90_110, epoch)

    return mae_100, visi


def LMDS_counting(input, w_fname, f_loc, args):
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

    return count100, count90_110, count95_105, kpoint, f_loc


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
