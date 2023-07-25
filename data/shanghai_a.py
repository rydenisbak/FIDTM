from torch.utils.data import Dataset
import albumentations as A
from tqdm import tqdm
from torchvision import transforms as tvt
import torch.nn.functional as F
import functools
import scipy.io as io
import numpy as np
import cv2


class ShanghaiADataset(Dataset):
    def __init__(self, data_file, mode, args, debug=False):
        super(ShanghaiADataset, self).__init__()
        cv2.setNumThreads(min(4, cv2.getNumberOfCPUs()))
        self.fnames = np.load(data_file).tolist()
        self.preload_data = args['preload_data']
        self.points, self.imgs = [], []
        for fname in tqdm(self.fnames, desc=f'load {data_file}'):
            gt_path = fname.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_')
            mat = io.loadmat(gt_path)
            points = np.float32(mat["image_info"][0][0][0][0][0])
            self.points.append(points)
            if self.preload_data:
                self.imgs.append(cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB))

        self.mode, self.debug = mode, debug
        self.crop_size = args['crop_size']
        self.augmentation, self.img_to_tensor, self.fidtm_to_tensor = self.create_transforms(args)
        self.val_scale = min(1., args.get('val_scale', 1.))  # do only downscale
        self.val_max_img_size = args.get('val_max_img_size', -1.)

    def create_transforms(self, args):
        assert self.crop_size % 16 == 0, 'crop size must be divisible by 16'
        augmentation = None
        if self.mode == 'train':
            if args['transforms_type'] == 'strong':
                augmentation = A.Compose([
                    A.PadIfNeeded(min_width=int(self.crop_size * 1.5),
                                  min_height=int(self.crop_size * 1.5),
                                  position='center'),
                    A.OneOrOther(
                        first=A.RandomSizedCrop(
                            min_max_height=(int(self.crop_size * 0.5), int(self.crop_size * 1.5)),
                            height=self.crop_size,
                            width=self.crop_size,
                            w2h_ratio=1.0,
                        ),
                        second=A.RandomCrop(height=self.crop_size, width=self.crop_size),
                        p=0.9,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ChannelShuffle(p=0.5),
                    A.RGBShift(r_shift_limit=127, g_shift_limit=127, b_shift_limit=127, p=0.5),
                    A.ToGray(p=0.05),
                    A.Rotate(border_mode=cv2.BORDER_CONSTANT, limit=45, interpolation=4, p=.4, value=0),
                    A.ChannelDropout(),
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
            elif args['transforms_type'] == 'medium':
                augmentation = A.Compose([
                    A.PadIfNeeded(min_width=int(self.crop_size * 1.3),
                                  min_height=int(self.crop_size * 1.3),
                                  position='center'),
                    A.OneOrOther(
                        first=A.RandomSizedCrop(
                            min_max_height=(int(self.crop_size * 0.7), int(self.crop_size * 1.3)),
                            height=self.crop_size,
                            width=self.crop_size,
                            w2h_ratio=1.0,
                        ),
                        second=A.RandomCrop(height=self.crop_size, width=self.crop_size),
                        p=0.9,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.RGBShift(r_shift_limit=127, g_shift_limit=127, b_shift_limit=127, p=0.7),
                    A.ToGray(p=0.05),
                    A.Rotate(border_mode=cv2.BORDER_CONSTANT, limit=25, interpolation=4, p=.4, value=0),
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
            elif args['transforms_type'] == 'medium_05':
                augmentation = A.Compose([
                    A.PadIfNeeded(min_width=int(self.crop_size * 1.1),
                                  min_height=int(self.crop_size * 1.1),
                                  position='center'),
                    A.OneOrOther(
                        first=A.RandomSizedCrop(
                            min_max_height=(int(self.crop_size * 0.5), int(self.crop_size * 1.1)),
                            height=self.crop_size,
                            width=self.crop_size,
                            w2h_ratio=1.0,
                        ),
                        second=A.RandomCrop(height=self.crop_size, width=self.crop_size),
                        p=0.8,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.RGBShift(r_shift_limit=127, g_shift_limit=127, b_shift_limit=127, p=0.7),
                    A.ToGray(p=0.05),
                    A.Rotate(border_mode=cv2.BORDER_CONSTANT, limit=25, interpolation=4, p=.4, value=0),
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
            elif args['transforms_type'] == 'medium_03':
                augmentation = A.Compose([
                    A.PadIfNeeded(min_width=int(self.crop_size * 1.1),
                                  min_height=int(self.crop_size * 1.1),
                                  position='center'),
                    A.OneOrOther(
                        first=A.RandomSizedCrop(
                            min_max_height=(int(self.crop_size * 0.3), int(self.crop_size * 1.1)),
                            height=self.crop_size,
                            width=self.crop_size,
                            w2h_ratio=1.0,
                        ),
                        second=A.RandomCrop(height=self.crop_size, width=self.crop_size),
                        p=0.8,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.RGBShift(r_shift_limit=127, g_shift_limit=127, b_shift_limit=127, p=0.7),
                    A.ToGray(p=0.05),
                    A.Rotate(border_mode=cv2.BORDER_CONSTANT, limit=25, interpolation=4, p=.4, value=0),
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
            elif args['transforms_type'] == 'easy':
                augmentation = A.Compose([
                    A.PadIfNeeded(min_width=int(self.crop_size * 1.3),
                                  min_height=int(self.crop_size * 1.3),
                                  position='center'),
                    A.OneOrOther(
                        first=A.RandomSizedCrop(
                            min_max_height=(int(self.crop_size * 0.7), int(self.crop_size * 1.3)),
                            height=self.crop_size,
                            width=self.crop_size,
                            w2h_ratio=1.0,
                        ),
                        second=A.RandomCrop(height=self.crop_size, width=self.crop_size),
                        p=0.5,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.RGBShift(r_shift_limit=127, g_shift_limit=127, b_shift_limit=127, p=0.5),
                    A.ToGray(p=0.05),
                    A.Rotate(border_mode=cv2.BORDER_CONSTANT, limit=15, interpolation=4, p=.2, value=0),
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
            elif args['transforms_type'] == 'zero':
                augmentation = A.Compose([
                    A.PadIfNeeded(min_width=int(self.crop_size * 1.3),
                                  min_height=int(self.crop_size * 1.3),
                                  position='center'),
                    A.OneOrOther(
                        first=A.RandomSizedCrop(
                            min_max_height=(int(self.crop_size * 0.7), int(self.crop_size * 1.3)),
                            height=self.crop_size,
                            width=self.crop_size,
                            w2h_ratio=1.0,
                        ),
                        second=A.RandomCrop(height=self.crop_size, width=self.crop_size),
                        p=0.5,
                    ),
                    A.HorizontalFlip(p=0.5),
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

        if args['wo_normalization']:
            img_to_tensor = lambda img: tvt.functional.to_tensor(img.astype(np.float32))
        else:
            img_to_tensor = tvt.Compose([
                tvt.ToTensor(),
                tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        fidtm_to_tensor = tvt.ToTensor()
        return augmentation, img_to_tensor, fidtm_to_tensor

    @staticmethod
    def generate_fidt_and_point_maps(img, img_points, check_invisible=True):
        height, width = img.shape[:-1]
        zeros_like = functools.partial(np.zeros, shape=(height, width), dtype=np.float32)

        if len(img_points) == 0:
            return zeros_like(), zeros_like()

        xy_coords = np.round(img_points).astype(int)
        if check_invisible:
            x_visible = (xy_coords[:, 0] >= 0) & (xy_coords[:, 0] < width)
            y_visible = (xy_coords[:, 1] >= 0) & (xy_coords[:, 1] < height)
            xy_coords = xy_coords[x_visible & y_visible]

        if len(xy_coords) == 0:
            return zeros_like(), zeros_like()

        # generate fidt map
        d_map = np.full(shape=[height, width], fill_value=255, dtype=np.uint8)
        d_map[xy_coords[:, 1], xy_coords[:, 0]] = 0
        distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0).astype(np.float64)
        distance_map = 1 / (1 + np.power(distance_map, 0.02 * distance_map + 0.75))
        distance_map[distance_map < 1e-2] = 0
        distance_map = np.float32(distance_map)

        # generate point map
        point_map = np.zeros(shape=(height, width), dtype=np.float32)
        point_map[xy_coords[:, 1], xy_coords[:, 0]] = 1

        return distance_map, point_map

    def __len__(self):
        return len(self.points)

    @staticmethod
    def remove_invisible_points(img, img_points):
        if len(img_points) == 0:
            return img_points
        height, width = img.shape[:-1]
        x_visible = (img_points[:, 0] >= 0) & (img_points[:, 0] < width)
        y_visible = (img_points[:, 1] >= 0) & (img_points[:, 1] < height)
        return img_points[x_visible & y_visible]

    def __getitem__(self, index):
        fname, points = self.fnames[index], np.copy(self.points[index])
        if self.preload_data:
            img = np.copy(self.imgs[index])
        else:
            img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

        points = self.remove_invisible_points(img, points)

        if self.augmentation is not None:
            transformed = self.augmentation(image=img, keypoints=points)
            img, points = transformed['image'], np.array(transformed['keypoints'], np.float32).reshape(-1, 2)

        if self.val_scale < 1 or self.val_max_img_size > 0:
            assert self.augmentation is None and self.mode != 'train'
            if self.val_scale < 1 and self.val_max_img_size > 0:
                AttributeError('choose only --val_scale or --val_max_img_size not simultaneously')
            if self.val_scale < 1:
                resize_scale = self.val_scale
            else:
                resize_scale = min(1, self.val_max_img_size / max(img.shape))

            if resize_scale < 1:
                img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
                points = np.round(points.astype(np.float64) * resize_scale).astype(np.int32)
        fidt_map, point_map = self.generate_fidt_and_point_maps(img, points)

        if self.debug:
            point_map = np.uint8(point_map * 255)
            return img, fidt_map, points, point_map

        img, fidt_map = self.img_to_tensor(img), self.fidtm_to_tensor(fidt_map)

        if self.mode != 'train':
            pad_y = (16 - img.shape[1] % 16) % 16
            pad_x = (16 - img.shape[2] % 16) % 16
            if pad_y + pad_x > 0:
                img = F.pad(img, [0, pad_x, 0, pad_y], value=0)
                fidt_map = F.pad(fidt_map, [0, pad_x, 0, pad_y], value=0)
                point_map = np.pad(point_map, [(0, pad_y), (0, pad_x)], mode='constant', constant_values=0)

        return fname, img, fidt_map, point_map


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    dataset_args = dict(
        data_file='../npydata/ShanghaiA_train.npy',
        mode='train',
        args=dict(
            transforms_type='strong',
            crop_size=256,
            preload_data=True,
        ),
        debug=True,
    )

    shanghaia_dataset = ShanghaiADataset(**dataset_args)
    n_epoch = 2
    for epoch in range(n_epoch):
        for index in tqdm(range(len(shanghaia_dataset)), desc='debug_dataset'):
            img, fidt_map, points, points_map = shanghaia_dataset[index]
            clear_img = np.copy(img)
            points_map = np.dstack([points_map, points_map, points_map])
            for (x, y) in points:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

            fidt_map = fidt_map / np.max(fidt_map) * 255
            fidt_map = fidt_map.astype(np.uint8)
            fidt_map = cv2.cvtColor(cv2.applyColorMap(fidt_map, 2), cv2.COLOR_BGR2RGB)

            vis_sample = np.vstack([
                np.hstack([clear_img, img]),
                np.hstack([fidt_map, points_map]),
            ])
            # plt.figure(figsize=(8, 8))
            # plt.axis('off')
            # plt.imshow(vis_sample)
            # plt.show()

            cv2.imwrite(f'/Users/den/Downloads/sha_aug_vis/{epoch}_{index:05}.png', vis_sample[..., ::-1])
