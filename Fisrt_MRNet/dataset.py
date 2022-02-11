import os
import glob
import paddle
from transforms import Compose, ToTensor


class Dataset(paddle.io.Dataset):
    def __init__(self, dataset_root=None, transforms=[ToTensor()]):
        if dataset_root is None:
            raise ValueError("dataset_root is None")
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)

        self.moire_list = glob.glob(os.path.join(self.dataset_root, "images", "*.jpg"))
        self.gt_list = glob.glob(os.path.join(self.dataset_root, "gts", "*.jpg"))

        self.moire_list.sort()
        self.gt_list.sort()
        assert len(self.moire_list) == len(self.gt_list)

    def __getitem__(self, index):
        moire = self.moire_list[index]
        gt = self.gt_list[index]
        return self.transforms(moire, gt)

    def __len__(self):
        return len(self.moire_list)


if __name__ == '__main__':
    dataset = Dataset(dataset_root="/Users/alex/Downloads/moire_competition_dataset_1206/moire_train_dataset")

    for d in dataset:
        pass
