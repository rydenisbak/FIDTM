from glob import glob
import os
from os import path as osp
import cv2
import h5py
import numpy as np
import scipy.io as io
from tqdm import tqdm

'''change your path'''
root = '/Users/den/Downloads/ShanghaiTech/'

part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')

path_sets = [part_A_train, part_A_test, part_B_train, part_B_test]

for path_set in path_sets:
    os.makedirs(path_set.replace('images', 'gt_fidt_map'), exist_ok=True)
    os.makedirs(path_set.replace('images', 'gt_show'), exist_ok=True)


img_paths = []
for path in path_sets:
    for img_path in glob(osp.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()


def fidt_generate1(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = (lamda * gt_data).astype(int)

    for (x, y) in gt:
        if x >= new_size[1] or y >= new_size[0] or min(x, y) < 0:
            continue
        d_map[y, x] = 0

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0).astype(np.float64)
    distance_map = 1 / (1 + np.power(distance_map, 0.02 * distance_map + 0.75))
    distance_map[distance_map < 1e-2] = 0
    return np.float32(distance_map)


for img_path in tqdm(img_paths):
    Img_data = cv2.imread(img_path)

    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    Gt_data = mat["image_info"][0][0][0][0][0]

    fidt_map1 = fidt_generate1(Img_data, Gt_data, 1)

    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for (gt_x, gt_y) in np.int64(Gt_data):
        if gt_x < Img_data.shape[1] and gt_y < Img_data.shape[0] and min(gt_x, gt_y) >= 0:
            kpoint[gt_y, gt_x] = 1

    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'gt_fidt_map'), 'w') as hf:
        hf['fidt_map'] = fidt_map1
        hf['kpoint'] = kpoint

    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)

    '''for visualization'''
    cv2.imwrite(img_path.replace('images', 'gt_show').replace('jpg', 'jpg'), fidt_map1)
