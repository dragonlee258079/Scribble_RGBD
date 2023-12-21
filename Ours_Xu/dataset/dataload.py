import torch.utils.data as data
from PIL import Image
from dataset.our_transforms import Compose

from dataset.preprocess import normalization
from dataset.general_augmentation import Scale, Random_Crop, Random_Flip


class BaseDataset(data.Dataset):
    def __init__(self, data_file, cfg, split='train'):
        super(BaseDataset, self).__init__()

        self.cfg = cfg

        file = open(data_file, 'rb')
        self.data = [f.strip().decode() for f in file.readlines()]

        self.split = split

        self.imgs = []
        self.depths = []
        self.gts = []
        self.masks = []

        self._read_data()

        self._augmentation()

    def _read_data(self):
        if self.split == 'train':
            for d in self.data:
                img, depth, gt, mask = d.split(' ')
                self.imgs.append(img)
                self.depths.append(depth)
                self.gts.append(gt)
                self.masks.append(mask)
        elif self.split == 'test':
            for d in self.data:
                img, depth = d.split(' ')[:2]
                self.imgs.append(img)
                self.depths.append(depth)

    def _augmentation(self):
        scale_size = self.cfg.DATASET.SCALE_SIZE
        target_size = self.cfg.DATASET.TARGET_SIZE
        if self.split == 'train':
            self.joint_transform = Compose([
                Scale(scale_size),
                Random_Crop(target_size),
                Random_Flip(),
            ])
        elif self.split == 'test':
            self.joint_transform = None
        else:
            raise Exception("Split not recognized")
        self.normalization = normalization(self.split, target_size)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        input_data = {}

        # image
        img_fname = self.imgs[idx]
        image = Image.open(img_fname).convert('RGB')
        input_data['image'] = image

        input_data['size'] = image.size
        input_data['image_name'] = img_fname.split('/')[-1]

        # depth
        depth_fname = self.depths[idx]
        depth = Image.open(depth_fname).convert('L')
        input_data['depth'] = depth

        if not self.split == 'test':
            gt_fname = self.gts[idx]
            gt = Image.open(gt_fname).convert('L')
            input_data['gt'] = gt

            mask_fname = self.masks[idx]
            mask = Image.open(mask_fname).convert('L')
            input_data['mask'] = mask

            # gray = Image.open(img_fname).convert('L')
            # input_data['gray'] = gray

            input_data = self.joint_transform(input_data)
            input_data = self.normalization(input_data)
        else:
            input_data = self.normalization(input_data)

        return input_data
