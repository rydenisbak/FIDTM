from tqdm import tqdm
import numpy as np
import cv2
from data.shanghai_a import ShanghaiADataset
from os import path as osp
from torch.utils.data import Dataset
import random


class ConcatDataset(Dataset):
    def __init__(self, datasets, train_roots, weights=None, dataset_len=None):
        super(ConcatDataset, self).__init__()
        print('Concat Dataset:')
        self.datasets = datasets
        self.n_datasets = len(datasets)

        if weights is None:
            weights = [len(ds) for ds in datasets]
        self.weights = weights

        for dataset, train_root, weight in zip(datasets, train_roots, weights):
            print(f'\t{osp.basename(train_root)} weight: {weight}, len: {len(dataset)}')

        if dataset_len is None:
            dataset_len = sum([len(ds) for ds in datasets])
        self.datasets_len = dataset_len
        print(f'\tdatasets_len: {self.datasets_len}')

    def __len__(self):
        return self.datasets_len

    def __getitem__(self, index):
        dataset_idx = random.choices(range(self.n_datasets), weights=self.weights, k=1)[0]
        dataset = self.datasets[dataset_idx]
        sample_idx = random.randint(0, len(dataset) - 1)
        return dataset[sample_idx]


class NWPUCrowdDataset(ShanghaiADataset):
    def __init__(self, data_root, mode, args, debug=False):
        super(ShanghaiADataset, self).__init__()
        cv2.setNumThreads(min(4, cv2.getNumberOfCPUs()))
        flist_file = osp.join(data_root, 'train.txt' if mode == 'train' else 'val.txt')
        self.fnames = []
        with open(flist_file, 'r') as f_id:
            for line in f_id.readlines():
                im_name = line.split(' ')[0] + '.jpg'
                self.fnames.append(osp.join(data_root, 'images_2048', im_name))

        self.preload_data = args['preload_data']
        self.points, self.imgs = [], []
        for fname in tqdm(self.fnames, desc=f'load {data_root}'):
            gt_path = fname.replace('/images_2048/', '/gt_npydata_2048/').replace('.jpg', '.npy')
            points = np.load(gt_path)
            self.points.append(points)
            if self.preload_data:
                self.imgs.append(cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB))

        self.mode, self.debug = mode, debug
        self.crop_size = args['crop_size']
        self.augmentation, self.img_to_tensor, self.fidtm_to_tensor = self.create_transforms(args)
        self.val_scale = min(1., args.get('val_scale', 1.))  # do only downscale
        self.val_max_img_size = args.get('val_max_img_size', -1.)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import os
    dataset_args = dict(
        data_root='/Users/den/Downloads/jhu_crowd_v2.0/',
        mode='train',
        args=dict(
            transforms_type='medium',
            crop_size=512,
            preload_data=False,
        ),
        debug=True,
    )
    out_folder = '/Users/den/Downloads/jhu_crowd_v2.0_vis'
    os.makedirs(out_folder, exist_ok=True)
    nwpu_crowd_dataset = NWPUCrowdDataset(**dataset_args)
    for index in tqdm(range(len(nwpu_crowd_dataset)), desc='debug_dataset'):
        img, fidt_map, points, points_map = nwpu_crowd_dataset[index]
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

        cv2.imwrite(osp.join(out_folder, f'{index:05}.png'), vis_sample[..., ::-1])
