import torch
import torch.utils.data as Data

import cv2
import os
import os.path as osp
import random
import numpy as np

__all__ = ['SirstAugDataset', 'IRSTD1kDataset', 'NUDTDataset', 'SirstDataset',
           'read_gray_pair', 'resize_pair', 'to_tensor_pair', 'list_png_names',
           'augumentation', 'PadImg', 'random_crop']


def list_png_names(directory):
    return [name for name in os.listdir(directory) if name.endswith('png')]


def read_gray_pair(img_path, mask_path):
    img, mask = cv2.imread(img_path, 0), cv2.imread(mask_path, 0)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    return img, mask


def resize_pair(img, mask, base_size):
    img = cv2.resize(img, (base_size, base_size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (base_size, base_size), interpolation=cv2.INTER_NEAREST)
    return img, mask


def to_tensor_pair(img, mask, base_size, pad=False):
    """Scale to [0, 1], reshape to (1, base_size, base_size) and convert to float tensors."""
    img = img.reshape(1, base_size, base_size) / 255.
    mask = mask.reshape(1, base_size, base_size)
    mask_max = np.max(mask)
    if mask_max > 0:
        mask = mask / mask_max
    if pad:
        img, mask = PadImg(img), PadImg(mask)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    mask = torch.from_numpy(mask).type(torch.FloatTensor)
    return img, mask


class PairedFolderDataset(Data.Dataset):
    '''
    Dataset for `<base_dir>/{trainval,test}/{images,masks}` layouts.

    Return: Single channel
    '''

    def __init__(self, base_dir, mode='train', base_size=256, resize=False, augment=False):
        assert mode in ['train', 'test']
        self.mode = mode
        self.data_dir = osp.join(base_dir, 'trainval' if mode == 'train' else 'test')
        self.base_size = base_size
        self.resize = resize
        self.names = list_png_names(osp.join(self.data_dir, 'images'))
        self.tranform = augumentation() if augment else None

    def __getitem__(self, i):
        name = self.names[i]
        img, mask = read_gray_pair(osp.join(self.data_dir, 'images', name),
                                   osp.join(self.data_dir, 'masks', name))
        if self.mode == 'train' and self.tranform is not None:
            img, mask = self.tranform(img, mask)
        if self.resize:
            img, mask = resize_pair(img, mask, self.base_size)
        return to_tensor_pair(img, mask, self.base_size)

    def __len__(self):
        return len(self.names)


class SirstAugDataset(PairedFolderDataset):
    def __init__(self, base_dir=r'/Users/tianfangzhang/Program/DATASETS/sirst_aug',
                 mode='train', base_size=256):
        super().__init__(base_dir, mode=mode, base_size=base_size, resize=False, augment=True)


class IRSTD1kDataset(PairedFolderDataset):
    def __init__(self, base_dir=r'D:/WFY/datasets/IRSTD-1k',
                 mode='train', base_size=256):
        super().__init__(base_dir, mode=mode, base_size=base_size, resize=True)


class NUDTDataset(PairedFolderDataset):
    def __init__(self, base_dir=r'D:/WFY/datasets/NUDT',
                 mode='train', base_size=256):
        super().__init__(base_dir, mode=mode, base_size=base_size, resize=True)


class SirstDataset(Data.Dataset):
    def __init__(self, base_dir=r'datasets/SIRSTv1',
                 mode='train', base_size=256):
        if mode == 'train':
            txtfile = 'trainval_v1.txt'
        elif mode == 'val' or mode == 'test':
            txtfile = 'test_v1.txt'
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'train', 'val', or 'test'.")

        self.list_dir = osp.join(base_dir, 'Splits', txtfile)
        self.imgs_dir = osp.join(base_dir, 'PNGImages')
        self.label_dir = osp.join(base_dir, 'SIRST/BinaryMask')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.base_size = base_size
        self.tranform = augumentation()

    def __getitem__(self, i):
        name = self.names[i]
        img, mask = read_gray_pair(osp.join(self.imgs_dir, name + '.png'),
                                   osp.join(self.label_dir, name + '_pixels0.png'))

        if self.mode == 'train':
            img, mask = self.tranform(img, mask)
            img, mask = resize_pair(img, mask, self.base_size)
            return to_tensor_pair(img, mask, self.base_size)

        img, mask = resize_pair(img, mask, self.base_size)
        return to_tensor_pair(img, mask, self.base_size, pad=True)

    def __len__(self):
        return len(self.names)


class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input.copy(), target.copy()

def PadImg(img, times=32):
    _, h, w = img.shape
    
    if not h % times == 0:
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0),(0, (w//times+1)*times-w)), mode='constant')
    return img


def random_crop(img, mask, patch_size, pos_prob=None):
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        h, w = img.shape

    cur_prob = random.random()
    if pos_prob == None or cur_prob > pos_prob or mask.max() == 0:
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
    else:
        loc = np.where(mask > 0)
        if len(loc[0]) <= 1:
            idx = 0
        else:
            idx = random.randint(0, len(loc[0]) - 1)
        h_start = random.randint(max(0, loc[0][idx] - patch_size), min(loc[0][idx], h - patch_size))
        w_start = random.randint(max(0, loc[1][idx] - patch_size), min(loc[1][idx], w - patch_size))

    h_end = h_start + patch_size
    w_end = w_start + patch_size
    img_patch = img[h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch
