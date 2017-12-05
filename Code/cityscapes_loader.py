import os
import torch
import numpy as np
from torch.utils import data
from PIL import Image

n_classes = 19
ignore_label = 255
root = 'dataset'

def obtain_dir(mode, root):
    # "mode" can be "train", "test", or "val".
    mask_path = os.path.join(root, 'gtFine', mode)
    img_path = os.path.join(root, 'leftImg8bit', mode)
    # The only difference between mask and image names are postfix.
    mask_postfix = '_gtFine_labelIds.png'
    # Get paths for all of 8-bit images and corresponding masks.
    items = []
    categories = os.listdir(img_path)
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)
    # Return a set of tuples.
    return items

class CityScapes(data.Dataset):
    def __init__(self, mode, root, transform=None, target_transform=None, joint_transform=None):
        self.imgs = obtain_dir(mode, root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        # Only care about 19 classes. See the dataset paper for details.
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        mask = torch.from_numpy(np.array(mask))
        mask = torch.Tensor.float(mask.view(1, mask.size(0), mask.size(1)))
        return img, mask

    def __len__(self):
        return len(self.imgs)