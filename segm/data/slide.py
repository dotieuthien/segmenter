import os

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
from PIL import Image
import scipy.io as sio

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


SLIDE_CATS_PATH = Path(__file__).parent / "config" / "slide.yml"


def read_mat_file(path, img_size):
    label = sio.loadmat(path)
    inst_map = label['inst_map']
    label_map = np.zeros_like(inst_map, dtype=np.uint8)

    nuclei_id = label['id']
    classes = label['class']

    bboxs = label['bbox']
    centroids = label['centroid']
    unique_values = np.unique(inst_map).tolist()[1:]
    nuclei_id = np.squeeze(nuclei_id).tolist()

    for value in unique_values:
        # Get the position of the corresponding value
        idx = nuclei_id.index(value)
        class_ = classes[idx]
        rows, cols = np.where(inst_map == value)
        label_map[rows, cols] = class_[0]

        # y1, y2, x1, x2 = bboxs[idx]
        # centroid = centroids[idx]

        # Display the image
        # fig, ax = plt.subplots()
        # ax.imshow(inst_map)
        # import matplotlib.patches as patches
        # rect = patches.Rectangle((x1, y1), y2 - y1, x2 - x1, linewidth=1, edgecolor='r', facecolor='none')
        # plt.plot(centroid[0], centroid[1], marker="o", markersize=1, markeredgecolor="red", markerfacecolor="green")
        # ax.add_patch(rect)
        # plt.show()

    label_map = cv2.resize(label_map, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return label_map


class SlideDataset(Dataset):
    def __init__(
            self,
            image_size=224,
            crop_size=224,
            split='train',
            normalization='vit',
            **kwargs,
    ):
        super(SlideDataset, self).__init__()

        self.root_dir = kwargs["dataset_dir"]
        self.image_size = image_size
        self.split = split
        self.normalization = normalization

        if split == "train":
            self.transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

        self.n_cls = 7

        # Get slide images
        images_list = natsorted(glob.glob(os.path.join(self.root_dir, 'images', '*')))

        # Get slide segment mask
        self.images_list = []
        self.masks_list = []

        self.weighted_loss = np.array([0] * self.n_cls, dtype=np.float64)

        for img_path in images_list:
            img_name = os.path.basename(img_path)[:-4]
            label_path = os.path.join(self.root_dir, 'annotations', img_name + '.mat')

            if os.path.exists(label_path):
                self.images_list.append(img_path)
                self.masks_list.append(label_path)

        #         # Read target label
        #         target = read_mat_file(label_path, self.image_size)
        #         uniques, counts = np.unique(target, return_counts=True)
        #         counts = max(counts) / counts
        #         self.weighted_loss[uniques] += counts

        # self.weighted_loss = self.weighted_loss / max(self.weighted_loss)

    @property
    def unwrapped(self):
        return self

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        im = np.array(Image.open(self.images_list[idx]), dtype=np.uint8)
        im = cv2.resize(im, (self.image_size, self.image_size))
        target = read_mat_file(self.masks_list[idx], self.image_size)

        # Transform
        im = self.transform(im)
        target = torch.from_numpy(target).squeeze(0)

        return dict(im=im, segmentation=target)


if __name__ == '__main__':
    read_mat_file('/home/hades/Desktop/segmenter/segm/data/enzo/annotations/consep_1.mat', 512)
