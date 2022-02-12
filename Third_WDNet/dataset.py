import os
import glob
import paddle
from transforms import Compose
import numpy as np

def random_crop(im1, im2, patch_size = 512):
    assert im1.shape == im2.shape, "Value Error!"

    _, h, w = im1.shape

    # random choose top coordinated for im1, im2 patch
    top = np.random.randint(h - patch_size + 1)
    left = np.random.randint(w - patch_size + 1)

    im1 = im1[:, top:top+patch_size, left:left+patch_size]
    im2 = im2[:, top:top+patch_size, left:left+patch_size]

    return im1, im2


def read_anno_file(path):
    data = []

    with open(path, 'r') as f:
        data = f.readlines()
        data = [v.rstrip('\n') for v in data]

    return data


class AIMDataset(paddle.io.Dataset):
    def __init__(self, dataset_root, transforms, random_crop=True):
        if dataset_root is None:
            raise ValueError("dataset_root is None")
            
        self.dataset_root = dataset_root

        if transforms is not None:
            self.transforms = Compose(transforms)
        else:
            self.transforms = None
        
        self.random_crop = random_crop

        self.lq_list = glob.glob(os.path.join(self.dataset_root, "ValidationMoire", "*.png"))
        self.gt_list = glob.glob(os.path.join(self.dataset_root, "ValidationClear", "*.png"))

        self.lq_list.sort()
        self.gt_list.sort()

    def __getitem__(self, index):
        lq = self.lq_list[index]
        gt = self.gt_list[index]

        if self.transforms is not None:
            lq, gt = self.transforms(lq, gt)
        
        if self.random_crop:
            lq, gt = random_crop(lq, gt, patch_size=512)
        
        return lq, gt

    
    def __len__(self):
        return len(self.lq_list)


class Dataset(paddle.io.Dataset):
    def __init__(self, dataset_root=None, transforms=None, val=False, test_mode=False, patch_size=512):
        if dataset_root is None:
            raise ValueError("dataset_root is None")

        self.dataset_root = dataset_root
        self.patch_size = patch_size

        if transforms is not None:
            self.transforms = Compose(transforms)
        else:
            self.transforms = None
        
        self.random_crop = True

        self.im1_list = glob.glob(os.path.join(self.dataset_root, "images", "*.jpg"))
        self.im2_list = glob.glob(os.path.join(self.dataset_root, "gts", "*.jpg"))

        self.with_val = val
        if val:
            val_partition = ["moire_train_{:05d}.jpg".format(v) for v in range(970, 1000)]

            im1_list = self.im1_list
            im2_list = self.im2_list

            self.im1_list = [v for v in im1_list if os.path.basename(v) not in val_partition]
            self.im2_list = [v for v in im2_list if os.path.basename(v) not in val_partition]

            self.im1_val_partition = [os.path.join(self.dataset_root, "images", v) for v in val_partition]
            self.im2_val_partition = [os.path.join(self.dataset_root, "gts", v) for v in val_partition]

        self.im1_list.sort()
        self.im2_list.sort()

        if val:
            self.im1_val_partition.sort()
            self.im2_val_partition.sort()
        
        self.test_mode = test_mode

        assert len(self.im1_list) == len(self.im2_list)


    def __getitem__(self, index):
        im1 = self.im1_list[index]
        im2 = self.im2_list[index]
        
        if self.transforms is not None:
            im1, im2 = self.transforms(im1, im2)
        
        if self.random_crop:
            im1, im2 = random_crop(im1, im2, self.patch_size)
        
        return im1, im2


    def __len__(self):
        return len(self.im1_list)


class TIPDataset(paddle.io.Dataset):
    def __init__(self, dataset_root, transforms, patch_size=512):
        if dataset_root is None:
            raise ValueError("dataset_root is None")
        self.dataset_root = dataset_root

        if transforms is not None:
            self.transforms = Compose(transforms)
        else:
            self.transforms = None
        
        self.random_crop = True
        self.patch_size = patch_size

        self.lq_list = glob.glob(os.path.join(self.dataset_root, "trainData", "source", "*.png"))
        self.gt_list = glob.glob(os.path.join(self.dataset_root, "trainData", "target", "*.png"))

        self.lq_list.sort()
        self.gt_list.sort()

    def __getitem__(self, index):
        lq = self.lq_list[index]
        gt = self.gt_list[index]
        
        if self.transforms is not None:
            lq, gt = self.transforms(lq, gt)
        
        if self.random_crop:
            lq, gt = random_crop(lq, gt, patch_size=self.patch_size)
        
        return lq, gt

    def __len__(self):
        return len(self.lq_list)
