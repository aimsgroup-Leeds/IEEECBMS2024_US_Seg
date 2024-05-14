#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 05 July 2023
@author: Edward Ellis
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import utils

from PIL import Image
from metrics import StreamSegMetrics
from datasets.US_Data import Segmentation_US_Data as USSegmentation
from utils import ext_transforms as et

import matplotlib
import time
import matplotlib.pyplot as plt
from backboned_unet.unet_PBA import Unet


def create_predFolder(ckpt):
    directoryName = 'US_Pred_test'
    ckpt_folder_path = ckpt.split('/')[:-1]
    
    if not os.path.exists(directoryName):
        os.mkdir(directoryName)

    if not os.path.exists(os.path.join(directoryName, *ckpt_folder_path)):
        os.makedirs(os.path.join(directoryName, *ckpt_folder_path))

    return os.path.join(directoryName, *ckpt_folder_path)


def detect_imgs(infolder, ext='.tif'):
    import os

    items = os.listdir(infolder)

    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")

    parser.add_argument("--data_root", type=str, default=r'/home/scs/Desktop/Eddy/US_Nerve',
                        help="path to Dataset")

    # parser.add_argument("--data_root", type=str, default='../../Data/BUSI',
    #                     help="path to Dataset")

    parser.add_argument("--dataset", type=str, default='US_Nerve',
                        choices=['BUSI', 'US_Nerve', 'voc', 'cityscapes'], help='Name of dataset')

    # Deeplab Options

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')

    parser.add_argument("--model", type=str, default='unet',
                        choices=['resnet-Unet', 'efficientnet-Unet', 'pvt_v2_b5-Unet', 'unet'], help='model name')

    parser.add_argument("--backbone", type=str, default='None',
                        choices=['resnet50', 'efficientnet_b0',
                                 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                                 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'pvt_v2_b5'],
                                  help='model name')

    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    parser.add_argument("--ckpt", type=str, default='US_Nerve_None_lr0.001/best_val_loss_None_US_Nerve_os16_US.pth',
                        help="checkpoint file")

    parser.add_argument("--gpu_id", type=str, default='3',
                        help="GPU ID")

    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    return parser

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        # et.ExtResize(size=opts.crop_size),
        et.ExtResize(size=384),
        # et.ExtRandomScale((0.5, 2.0)),
        # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        # et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
        et.ExtRandomVerticalFlip(),
        et.ExtRandomHorizontalFlip(),
        et.ExtRandomDiagonalFlip(),
        et.ExtRandomRotation(degrees=15),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(256),
            # et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtResize(384),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    test_dst = USSegmentation(root=opts.data_root, dataset=opts.dataset, image_set='test', transform=val_transform)

    return test_dst


def mymodel():
    """
    Returns
    -------
    model : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    """
    opts = get_argparser().parse_args()
    # ---> explicit class number
    opts.num_classes = 2

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Set up model      
    if opts.model.lower()  == 'unet':
        from network.unet import UNet
        model = UNet(n_channels=3, n_classes=opts.num_classes)
        opts.backbone = 'None'

    elif opts.model == 'resnet-Unet':
        # net = Unet(backbone_name='densenet121', classes=2)
        model = Unet(backbone_name=[opts.backbone], pretrained=True, classes=opts.num_classes,
                     decoder_filters=(512, 256, 128, 64, 32))

    elif opts.model == 'efficientnet-Unet':
        model = Unet(backbone_name=[opts.backbone], pretrained=True, classes=opts.num_classes,
                     decoder_filters=(128, 64, 32, 16, 8))
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('(efficientNet number of trainable parameters:', pytorch_total_params)

    elif opts.model == 'pvt_v2_b5-Unet':
        model = Unet(backbone_name=[opts.backbone], pretrained=True, classes=opts.num_classes,
                     decoder_filters=(128, 64, 32, 16), final_upsampling=True)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('PVT-Tiny: Number of trainable parameters:', pytorch_total_params)

    elif opts.model == 'unet':
        from network.unet import UNet
        model = UNet(n_channels=3, n_classes=1, bilinear=True)

    # model = nn.DataParallel(model)
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints', opts.ckpt)
    checkpoint = torch.load(checkpoint_dir, map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint["model_state"])
    state_dict = checkpoint['model_state']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    #
    # for k, v in state_dict.items():
    #     if 'module' not in k:
    #         k = 'module.' + k
    #     else:
    #         k = k.replace('features.module.', 'module.features.')
    #     new_state_dict[k] = v

    model.load_state_dict(state_dict)

    model = nn.DataParallel(model)
    model.eval()
    return model, device


if __name__ == '__main__':

    model, device = mymodel()

    opts = get_argparser().parse_args()
    test_dst = get_dataset(opts)

    saveDir = create_predFolder(opts.ckpt)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    Time_file = open(saveDir + '/' + "timeElaspsed" + '.txt', mode='w')
    Metrics_file = open(saveDir + '/' + "Metrics" + '.txt', mode='w')

    timeappend = []

    split_f = os.path.join(opts.data_root, 'test.txt')
    with open(os.path.join(split_f), "r") as f:
        file_names = [x.strip() for x in f.readlines()]

    test_loader = data.DataLoader(
        test_dst, batch_size=1, shuffle=False, num_workers=2)

    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
    
    print("Model device:", next(model.parameters()).device)

    total_time = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            # size=images.shape

            # Create new events for each iteration
            if device == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                predicted_outputs = model(images)
                end.record()
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)
                print(elapsed_time)
                timeappend.append(elapsed_time)

            else:
                start_time = time.time()

                predicted_outputs = model(images)

                end_time = time.time()

                elapsed_time = end_time - start_time
                print("Elapsed Time:", elapsed_time)
                timeappend.append(elapsed_time)
                
            total_time += elapsed_time

            preds = predicted_outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            image = images[0]  # 0 added as batch size is 1
            target = targets[0]
            pred = preds[0]

            metrics.update(target, pred)

            image = image.detach().cpu().numpy()
            image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)

            target = (target * 255.0).astype(np.uint8)
            pred = (pred * 255.0).astype(np.uint8)

            Image.fromarray(image).save(saveDir + '/' + file_names[i] + '_image.png')
            Image.fromarray(target).save(saveDir + '/' + file_names[i] + '_target.png')
            Image.fromarray(pred).save(saveDir + '/' + file_names[i] + '_pred.png')

            fig = plt.figure()
            plt.imshow(image)
            plt.axis('off')
            plt.imshow(pred, alpha=0.7)
            ax = plt.gca()
            ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            plt.savefig(saveDir + '/' + file_names[i] + '_overlay.png', bbox_inches='tight', pad_inches=0)
            plt.close()

            Time_file.write('%s -----> %s \n' %
                            (file_names[i], elapsed_time))

        Time_file.close()
        
        average_time_per_iteration = total_time / len(test_loader) 
        fps = 1 / average_time_per_iteration
        print("Average FPS:", fps)
        
        for key, value in metrics.get_results(Eval=True).items():
            if key == "Top 5 File indices":
                top_5_formatted = ", ".join(str(idx) for idx in value)
                line = f"Top 5 File indices: {{ {top_5_formatted} }}"
            elif key == "Bottom 5 file indices":
                bottom_5_formatted = ", ".join(str(idx) for idx in value)
                line = f"Bottom 5 File indices: {{ {bottom_5_formatted} }}"
            if isinstance(value, dict):
                formatted_value = ', '.join(f'{k}: {v:.4f}' for k, v in value.items())
                line = f'{key}: {{ {formatted_value} }}'
            else:
                line = f'{key}: {value:.4f}' if isinstance(value, float) else f'{key}: {value}'
            Metrics_file.write(line + '\n')
            
        # Add FPS to the metrics file
        Metrics_file.write(f'Average FPS: {fps:.2f}\n')
        Metrics_file.close()
