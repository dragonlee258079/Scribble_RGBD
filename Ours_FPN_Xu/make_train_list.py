import os

file_dir = "./data/train/train_list_xu.txt"

data_root = "/disk2/lilong/WRGBD/Dataset/Train_Xu"

file = open(file_dir, 'a')

img_list = os.listdir(os.path.join(data_root, 'img'))

for img_name in img_list:
    print(img_name)
    img_dir = os.path.join(data_root, 'img', img_name)
    dep_dir = os.path.join(data_root, 'depth', img_name[:-4]+'.png')
    gt_dir = os.path.join(data_root, 'gt', img_name[:-4]+'.png')
    mask_dir = os.path.join(data_root, 'mask', img_name[:-4]+'.png')
    line = "{} {} {} {}\n".format(img_dir, dep_dir, gt_dir, mask_dir)
    file.write(line)

file.close()