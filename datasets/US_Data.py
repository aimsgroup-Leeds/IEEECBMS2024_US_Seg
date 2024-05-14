import os
import sys

current_dir = os.getcwd()
utils_dir = current_dir.replace("/datasets", "/utils")
sys.path.append(utils_dir)  # Add parent folder to the sys.path

import torch.utils.data as data
import numpy as np
from PIL import Image
from utils import ext_transforms as et
# import ext_transforms as et
# from utils import ext_transforms as et
# import ext_transforms as et
import argparse

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        # et.ExtResize(size=opts.crop_size),
        # et.ExtResize(size=opts.crop_size),
        # et.ExtRandomScale((0.5, 2.0)),
        # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        # et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    train_dst = Segmentation_US_Data(root=opts.data_root,
                                     image_set='train', transform=train_transform)
    return train_dst

def voc_cmap(N=256, normalized=True):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap

    return cmap


# def get_cross_val_datasets(root, dataset, num_folds, transform):
#     fold_datasets = []

#     for i in range(num_folds):
#         train_dataset = Segmentation_US_Data(root=root, dataset=dataset, image_set=f'cross_val_train_{i}', transform=transform)
#         val_dataset = Segmentation_US_Data(root=root, dataset=dataset, image_set=f'cross_val_val_{i}', transform=transform)
#         fold_datasets.append((train_dataset, val_dataset))

#     return fold_datasets


class Segmentation_US_Data(data.Dataset):
    """
    Args:
        root (string): Root directory of the US Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()

    def __init__(self,
                 root,
                 dataset='BUSI',
                 image_set='train',
                 cross_val=False,
                 num_folds=None,
                 fold_idx=None,
                 transform=None):

        self.root = os.path.expanduser(root)

        self.transform = transform

        self.image_set = image_set
        
        
        if cross_val:
            
            if dataset == 'BUSI':
                mask_dir = os.path.join(self.root, 'combined_maskGT')
            else:
                mask_dir = os.path.join(self.root, 'mask_ground_truth')
            assert os.path.exists(
                mask_dir), "Mask directory not found"
            
            if fold_idx == 0:
                if image_set == 'train':
                    split_f = os.path.join(self.root, 'fold_1','fold_1_train.txt')
                elif image_set == 'val':
                    split_f = os.path.join(self.root, 'fold_1','fold_1_val.txt')
                    
            if fold_idx == 1:
                if image_set == 'train':
                    split_f = os.path.join(self.root, 'fold_2','fold_2_train.txt')
                elif image_set == 'val':
                    split_f = os.path.join(self.root, 'fold_2','fold_2_val.txt')
                    
            if fold_idx == 2:
                if image_set == 'train':
                    split_f = os.path.join(self.root, 'fold_3','fold_3_train.txt')
                elif image_set == 'val':
                    split_f = os.path.join(self.root, 'fold_3','fold_3_val.txt')
                          
            if fold_idx == 3:
                if image_set == 'train':
                    split_f = os.path.join(self.root, 'fold_4','fold_4_train.txt')
                
                elif image_set == 'val':
                    split_f = os.path.join(self.root, 'fold_4','fold_4_val.txt')
                    
            if fold_idx == 4:
                if image_set == 'train':
                    split_f = os.path.join(self.root, 'fold_5','fold_5_train.txt')
                    
                elif image_set == 'val':
                    split_f = os.path.join(self.root, 'fold_5','fold_5_val.txt')
                    
                    
        else:

            if image_set == 'train':
                # not used!
                if dataset == 'BUSI':
                    mask_dir = os.path.join(self.root, 'combined_maskGT')
                else:
                    mask_dir = os.path.join(self.root, 'mask_ground_truth')
                assert os.path.exists(
                    mask_dir), "Mask directory not found"
                split_f = os.path.join(self.root, 'train.txt')  # './datasets/data/train_aug.txt'
                print('Train file:', split_f)
    
            elif image_set == 'val':
                if dataset == 'BUSI':
                    mask_dir = os.path.join(self.root, 'combined_maskGT')
                else:
                    mask_dir = os.path.join(self.root, 'mask_ground_truth')
                assert os.path.exists(
                    mask_dir), "Mask directory not found"
                split_f = os.path.join(self.root, 'val.txt')           
    
            elif image_set == 'test':
                if dataset == 'BUSI':
                    mask_dir = os.path.join(self.root, 'combined_maskGT')
                else:
                    mask_dir = os.path.join(self.root, 'mask_ground_truth')
                assert os.path.exists(
                    mask_dir), "Mask directory not found"
                split_f = os.path.join(self.root, 'test.txt')
    
            else:
                raise ValueError('Image set chosen not recognised')
            

        with open(os.path.join(split_f), "r") as f:
                file_names = [x.strip() for x in f.readlines()]
    
        image_dir = os.path.join(self.root, 'img')

        if dataset == 'BUSI':
            mask_dir = os.path.join(self.root, 'combined_maskGT')
            self.images = [os.path.join(image_dir, x + ".png") for x in file_names]
            # self.masks = [os.path.join(mask_dir, x + "_BE.tif") for x in file_names]
            self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        else:
            mask_dir = os.path.join(self.root, 'mask_ground_truth')
            self.images = [os.path.join(image_dir, x + ".tif") for x in file_names]
            # self.masks = [os.path.join(mask_dir, x + "_BE.tif") for x in file_names]
            self.masks = [os.path.join(mask_dir, x + "_mask.tif") for x in file_names]

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB').resize((256, 256), resample=0)

        target = Image.open(self.masks[index]).convert('RGB').resize((256, 256), resample=0)

        # if color ---> np.asarray(target)[:,:,0]
        target = Image.fromarray((np.asarray(target)[:, :, 0].astype('float32') / 255.0).astype('uint8'))
        # target = Image.fromarray((np.asarray(target)[:,:,0].astype('float32')/255.0))

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

def get_argparser():
    parser = argparse.ArgumentParser()
    return parser


if __name__ == '__main__':
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    from torch.utils import data

    opts = get_argparser()
    opts.data_root = r'../../../Data/BUSI'
    # opts.data_root = r'C:\Users\ekael\Mirror\PhD\Project\Datasets\US_Nerve'
    opts.dataset = 'BUSI'
    opts.crop_val = False
    train_dst = get_dataset(opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=1, shuffle=True, num_workers=2)
    print("Train set: %d" % (len(train_dst)))

    for (images, labels) in train_loader:
        print(images.shape)
        print(labels.shape)
        # images = images.to(device, dtype=torch.float32)
        # labels = labels.to(device, dtype=torch.long)

    # plt.imshow(images[0,:,:,:].permute(1,2,0))
    # plt.imshow(labels[0,:,:])
