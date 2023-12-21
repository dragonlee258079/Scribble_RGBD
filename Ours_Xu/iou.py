import os
import cv2
import numpy as np


gt_dir = '/data1/lilong/Weakly_RGBD/Datatset/Train/full_gt'
depth_dir = '/data1/lilong/Weakly_RGBD/Datatset/Train/depth_binary'

save_dir = '/data1/lilong/Weakly_RGBD/Datatset/Train/depth_binary_reverse'

imgs = os.listdir(depth_dir)

for i in range(len(imgs)):

    print('\r{}/{}'.format(i, len(imgs)), end="", flush=True)

    img_name = imgs[i]

    # if img_name != 'DUT-RGBD_0298.png':
    #     continue

    f_scr = cv2.imread(os.path.join(gt_dir, img_name), 1) / 255.
    depth = cv2.imread(os.path.join(depth_dir, img_name), 1) / 255.
    if f_scr.shape != depth.shape:
        depth = cv2.resize(depth, (f_scr.shape[1], f_scr.shape[0]))
    sum = f_scr + depth
    iou_ = np.sum((sum == 2).astype(np.float)) / np.sum(f_scr)
    if iou_ < 0.5:
        depth_ = np.zeros_like(depth)
        depth_[depth == 0] = 255
        cv2.imwrite(os.path.join(save_dir, img_name), depth_)
        continue
    cv2.imwrite(os.path.join(save_dir, img_name), depth*255)
