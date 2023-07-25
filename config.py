import argparse

parser = argparse.ArgumentParser(description='FIDTM')

parser.add_argument('--save_path', type=str, default='save_file/A_baseline',
                    help='save checkpoint directory')
parser.add_argument('--workers', type=int, default=16,
                    help='load data workers')
parser.add_argument('--print_freq', type=int, default=10,
                    help='print frequency')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch for training')
parser.add_argument('--epochs', type=int, default=3000,
                    help='end epoch for training')
parser.add_argument('--pre', type=str, default=None,
                    help='pre-trained model directory')
# parser.add_argument('--pre', type=str, default='./model_best_qnrf.pth',
#                     help='pre-trained model directory')


parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--crop_size', type=int, default=256,
                    help='crop size for training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--best_pred', type=int, default=1e5,
                    help='best pred')
parser.add_argument('--gpu_id', type=str, default='1',
                    help='gpu id')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--preload_data', action='store_true', default=False,
                    help='preload data. ')
parser.add_argument('--only_validate', action='store_true', default=False)
parser.add_argument('--visual', type=bool, default=False,
                    help='visual for bounding box. ')
parser.add_argument('--normalize_mse', action='store_true', default=False)
parser.add_argument('--norm_eval', action='store_true', default=False)
parser.add_argument('--norm_eval_encoder', action='store_true', default=False)
parser.add_argument('--loss_type', action='store', default='MSE', choices=['MSE', 'MSE_SSIM'])
parser.add_argument('--ssim_coefficient', action='store', default=1., type=float)
parser.add_argument('--transforms_type', action='store', default='ordinary',
                    choices=['ordinary', 'strong', 'medium', 'easy', 'zero', 'medium_03', 'medium_05'])
parser.add_argument('--model_type', action='store', default='hrnet')
parser.add_argument('--head_type', action='store', default='hrnet', choices=['hrnet', 'flower'])

parser.add_argument('--lr_scheduler', action='store', default='multistep',
                    choices=['multistep', 'cosine', 'warmup_cosine'])
parser.add_argument('--lr_steps', action='store', type=str, default='1000000000')
parser.add_argument('--warmup_epochs', action='store', type=int, default=100)
parser.add_argument('--epochs_wo_logs', action='store', type=int, default=25)
parser.add_argument('--optimizer', action='store', default='adam', choices=['sgd', 'adam'])
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--load_checkpoint', action='store', default=None, type=str)


# data
parser.add_argument('--dataset', type=str, default='ShanghaiA', help='choice train dataset')
parser.add_argument('--nwpu_root', type=str, default='/ssd/data/crowd_counting/NWPU_localization')
parser.add_argument('--val_roots', action='append')
parser.add_argument('--val_scale', type=float, default=1)
parser.add_argument('--val_max_img_size', type=float, default=-1)
parser.add_argument('--train_roots', action='append')
parser.add_argument('--train_weights', action='append', default=None, type=float)

parser.add_argument('--onnx_file', action='store', type=str, default=None)
parser.add_argument('--to_avx2', action='store_true', default=False)

# flower args
parser.add_argument('--flower_plan', action='store', default=None)
parser.add_argument('--wo_normalization', action='store_true', default=False)

# face engine args
parser.add_argument('--fe_data_path', action='store', default=None)

'''video demo'''
parser.add_argument('--video_path', type=str, default=None,
                    help='input video path ')

args = parser.parse_args()
return_args = parser.parse_args()
