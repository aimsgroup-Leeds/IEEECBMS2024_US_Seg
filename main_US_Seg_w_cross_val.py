# from tqdm import tqdm
import network

import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data

from torch.utils.tensorboard import SummaryWriter

from datasets.US_Data import Segmentation_US_Data as USSegmentation

from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize
from backboned_unet.unet_PBA import Unet

from sklearn.model_selection import StratifiedKFold



def plotInference(imgs, depth):
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(resize(imgs, (384, 384)))
    f.add_subplot(1, 2, 2)
    depth_masked = resize(depth, (384, 384))
    plt.imshow(depth_masked)
    # plt.show(block=True)
    return f


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default= '../../Data/BUSI',
                      help="path to Dataset") # Eddy pc
    #
    # parser.add_argument("--data_root", type=str, default= '../../Data/US_Nerve', # ARC path
    #                     help="path to Dataset")

    parser.add_argument("--dataType", type=str, default='US',
                        help="path to Dataset")

    parser.add_argument("--dataset", type=str, default='BUSI',
                        choices=['BUSI', 'US_Nerve', 'voc', 'cityscapes'], help='Name of dataset')

    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")

    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    parser.add_argument("--model", type=str, default='efficientnet-Unet',
                        choices=['Unet', 'resnet-Unet', 'efficientnet-Unet', 'pvt_v2_b-Unet', 'unet'], help='model name')

    parser.add_argument("--backbone", type=str, default='efficientnet_b7',
                        choices=['None', 'resnet50', 'efficientnet_b0',
                                 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                                 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'pvt_v2_b2',
                                 'pvt_v2_b5'],
                        help='model name')

    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results_US_Seg\"")
    parser.add_argument("--total_itrs", type=int, default=70e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--lr_min", type=int, default=1e-6)
    parser.add_argument("--step_size", type=int, default=10000)

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')

    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=True)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--early_stopping", action='store_true', default = True)
    parser.add_argument("--cross_val", action='store_true', default = True)


    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts, fold=None):
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
            et.ExtResize(384),
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
           
    if opts.cross_val:
        
        num_folds = 5 # only setup for 5 fold cross val
        
        if fold is None:
            raise ValueError("Cross-validation fold not specified.")
        if fold < 0 or fold >= num_folds:
            raise ValueError("Invalid cross-validation fold index.")

        train_dst = USSegmentation(root=opts.data_root, dataset=opts.dataset, image_set='train', cross_val=True, fold_idx=fold, transform=train_transform)
        val_dst = USSegmentation(root=opts.data_root, dataset=opts.dataset, image_set='val', cross_val=True, fold_idx=fold, transform=val_transform)

    else: 

        train_dst = USSegmentation(root=opts.data_root, dataset=opts.dataset, image_set='train', transform=train_transform)
        val_dst = USSegmentation(root=opts.data_root, dataset=opts.dataset, image_set='val', transform=val_transform)

    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    directoryName = 'results_%s_%s' % (opts.dataType, opts.backbone)
    if opts.save_val_results:
        if not os.path.exists(directoryName):
            os.mkdir(directoryName)
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    val_running_loss = 0
    val_num_minibatch = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            val_num_minibatch += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            # print('print target', targets[0].shape())
            # Val_loss
            if opts.loss_type == 'focal_loss':
                criterion = utils.FocalLoss(ignore_index=255, size_average=True)
            elif opts.loss_type == 'cross_entropy':
                criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
            val_loss = criterion(outputs, labels)

            np_val_loss = val_loss.detach().cpu().numpy()

            val_running_loss += np_val_loss

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for j in range(len(images)):
                    image = images[j].detach().cpu().numpy()
                    target = targets[j]
                    pred = preds[j]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    # target = loader.dataset.decode_target(target).astype(np.uint8)
                    # pred = loader.dataset.decode_target(pred).astype(np.uint8)
                    target = (target * 255.0).astype(np.uint8)
                    pred = (pred * 255.0).astype(np.uint8)

                    Image.fromarray(image).save(directoryName + '/%d_image.png' % img_id)
                    Image.fromarray(target).save(directoryName + '/%d_target.png' % img_id)
                    Image.fromarray(pred).save(directoryName + '/%d_pred.png' % img_id)

                    #fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig(directoryName + '/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return val_running_loss / val_num_minibatch, score, ret_samples


def main():
    opts = get_argparser().parse_args()
    
    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
   
    if opts.cross_val:
        
        num_folds = 5  # only set up for 5 folds

        
        for fold_idx in range(num_folds):
            
            print(fold_idx)
        
            print(f"Fold {fold_idx+1}/{num_folds}")
            
            print(fold_idx)
            
            train_dst, val_dst = get_dataset(opts, fold=fold_idx)
            train_loader = data.DataLoader(
                train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
            val_loader = data.DataLoader(
                val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
            print("Dataset: %s, Train set: %d, Val set: %d" %
                  (opts.dataset, len(train_dst), len(val_dst)))
            
            # Set up model               
            if opts.model.lower()  == 'unet':
                from network.unet import UNet
                model = UNet(n_channels=3, n_classes=opts.num_classes)
                opts.backbone = 'None'

            elif opts.model == 'resnet-Unet':
                # from backboned_unet.backboned_unet import Unet
                # net = Unet(backbone_name='densenet121', classes=2)
                model = Unet(backbone_name=[opts.backbone], pretrained=True, classes=opts.num_classes,
                             decoder_filters=(512, 256, 128, 64, 32))

            elif opts.model == 'efficientnet-Unet':
                # from backboned_unet.backboned_unet import Unet
                # net = Unet(backbone_name='densenet121', classes=2)
                model = Unet(backbone_name=[opts.backbone], pretrained=True, classes=opts.num_classes,
                             decoder_filters=(128, 64, 32, 16, 8))
                pytorch_total_params = sum(p.numel() for p in model.parameters())
                print('efficientNet: Number of trainable parameters:', pytorch_total_params)
                
            elif opts.model == 'pvt_v2_b-Unet':
                model = Unet(backbone_name = [opts.backbone], pretrained=True, classes=opts.num_classes,
                             decoder_filters = (128, 64, 32, 16), final_upsampling=True)
                pytorch_total_params = sum(p.numel() for p in model.parameters())
                print('PVT: Number of trainable parameters:', pytorch_total_params)

            # Set up metrics
            metrics = StreamSegMetrics(opts.num_classes)
              
            if opts.model == 'UNet':
                optimizer = torch.optim.Adam(params=model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
                
            elif opts.model == 'resnet-Unet' or 'efficientnet_Unet' or 'pvt_v2_b-Unet':
                optimizer = torch.optim.Adam([{'params': network.get_pretrained_parameters(), 'lr': 1e-5},
                                              {'params': network.get_random_initialized_parameters()}], lr=1e-4)

            else:
                optimizer = torch.optim.SGD(params=[
                    {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
                    {'params': model.classifier.parameters(), 'lr': opts.lr},
                ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

            if opts.lr_policy == 'poly':
                scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
            elif opts.lr_policy == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

            # Set up criterion
            if opts.loss_type == 'focal_loss':
                criterion = utils.FocalLoss(ignore_index=255, size_average=True)
            elif opts.loss_type == 'cross_entropy':
                criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

            print('no. of trainable hyperparameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

            def save_ckpt(path):
                """ save current model
                """
                torch.save({
                    "cur_itrs": cur_itrs,
                    "cur_epochs": cur_epochs,
                    "best_val_loss": best_val_loss,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "meanIOU_score": best_meanIOU_score,
                    "Dice_score": best_dice_score,
                }, path)
                print("Model saved as %s" % path)

            utils.mkdir(f'checkpoints/{opts.dataset}_{opts.backbone}_lr{opts.lr}')
            
            # Restore
            best_meanIOU_score = 0.0
            best_dice_score = 0.0
            cur_itrs = 0
            best_val_loss = float('inf')

            if opts.ckpt is not None and os.path.isfile(os.path.join(os.getcwd(), 'checkpoints', opts.ckpt)):
                # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
                checkpoint = torch.load(os.path.join(os.getcwd(), 'checkpoints', opts.ckpt), map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint["model_state"])
                model = nn.DataParallel(model)
                model = model.to(device)
                if opts.continue_training:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    scheduler.load_state_dict(checkpoint["scheduler_state"])
                    cur_itrs = checkpoint["cur_itrs"]
                    cur_epochs = checkpoint["cur_epochs"]
                    best_meanIOU_score = checkpoint['meanIOU_score']
                    best_dice_score = checkpoint['Dice_score']
                    best_val_loss = checkpoint['best_val_loss']
                    # opts.lr = checkpoint["scheduler_state"]['_last_lr'][0]
                    print("Training state restored from %s" % opts.ckpt)
                print("Model restored from %s" % opts.ckpt)
                del checkpoint  # free memory
            else:
                print("[!] Retrain")
                model = nn.DataParallel(model)
                model = model.to(device)

            # ==========   Train Loop   ==========#
            vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                              np.int32) if opts.enable_vis else None  # sample idxs for visualization
            denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

            # setting up tensorboard            
            writer = SummaryWriter(log_dir='logs/cross_val_fold_{}/cross_val_Fold{}_{}_{}_lr{}'.format(fold_idx, fold_idx, opts.dataset, opts.backbone, opts.lr))

            # # Initialize early stopping variables
            if opts.early_stopping:
                # stopping_threshold = 0.5  # threshold between 0 and 1, determining what percentage increase the val_loss needs to activate stopping criteria
                min_delta = 0.01
                patience = 30  # used to indicate how many iterations val_loss is above threshold before stopping
                early_stopping_counter = 0

            interval_loss = 0
            while True:  # cur_itrs < opts.total_itrs:
                # =====  Train  =====
                model.train()
                running_loss = 0
                num_minibatch = 0
                for (images, labels) in train_loader:
                    cur_itrs += 1
                    num_minibatch += 1
                    cur_epochs = cur_itrs // len(train_loader)  # Update current epoch based on total iterations

                    images = images.to(device, dtype=torch.float32)
                    labels = labels.to(device, dtype=torch.long)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    np_loss = loss.detach().cpu().numpy()

                    running_loss += np_loss
                    interval_loss += np_loss
                    if vis is not None:
                        vis.vis_scalar('Loss', cur_itrs, np_loss)

                    if cur_itrs % 10 == 0:
                        interval_loss = interval_loss / 10
                        print("Epoch %d, Itrs %d/%d, Loss=%f" %
                              (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))

                        # Write the interval loss for the epoch to TensorBoard
                        writer.add_scalar('Interval_loss', interval_loss, cur_epochs, cur_itrs)

                        interval_loss = 0.0

                    if cur_itrs % opts.val_interval == 0:
                        save_ckpt(f'checkpoints/{opts.dataset}_{opts.backbone}_lr{opts.lr}/latest_%s_%s_os%d_%s_fold{fold_idx}.pth' %
                                  (opts.backbone, opts.dataset, opts.output_stride, opts.dataType))
                        print("intermediate validation...")
                        model.eval()
                        val_loss, val_score, ret_samples = validate(
                            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                            ret_samples_ids=vis_sample_id)
                        
                        print(scheduler.get_lr())

                        if val_loss < best_val_loss: # saving lowest val_loss model:
                            best_val_loss = val_loss
                            save_ckpt(f'checkpoints/{opts.dataset}_{opts.backbone}_lr{opts.lr}/best_val_loss_%s_%s_os%d_%s_fold{fold_idx}.pth' %
                                      (opts.backbone, opts.dataset, opts.output_stride, opts.dataType))

                        if val_score['Mean IoU'] > best_meanIOU_score:  # save best model
                            best_meanIOU_score = val_score['Mean IoU']
                            save_ckpt(f'checkpoints/{opts.dataset}_{opts.backbone}_lr{opts.lr}/meanIOUBest_%s_%s_os%d_%s_fold{fold_idx}.pth' %
                                      (opts.backbone, opts.dataset, opts.output_stride, opts.dataType))

                        if val_score['Avg Dice Coefficient'] > best_dice_score:  # save best model
                            best_dice_score = val_score['Mean IoU']
                            save_ckpt(f'checkpoints/{opts.dataset}_{opts.backbone}_lr{opts.lr}/AvgDiceScoreBest%s_%s_os%d_%s_fold{fold_idx}.pth' %
                                      (opts.backbone, opts.dataset, opts.output_stride, opts.dataType))


                        if vis is not None:  # visualize validation score and samples
                            vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                            vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                            vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                            for k, (img, target, lbl) in enumerate(ret_samples):
                                img = (denorm(img) * 255).astype(np.uint8)
                                target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                                lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                                concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                                vis.vis_image('Sample %d' % k, concat_img)
                        model.train()
                    scheduler.step()
               
                save_ckpt(f'checkpoints/{opts.dataset}_{opts.backbone}_lr{opts.lr}/latest_%s_%s_os%d_%s_fold{fold_idx}.pth' %
                          (opts.backbone, opts.dataset, opts.output_stride, opts.dataType))
                
                print("validation...")
                model.eval()
                val_loss, val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))

                writer.add_scalar('Loss/val', val_loss, cur_epochs)

                # Added flattened_metrics to be able to add metrics to tensorboard
                flattened_metrics = {}
                for key, value in val_score.items():
                    if isinstance(value, dict):
                        # Flatten the nested dictionary
                        for k, v in value.items():
                            tag = f"{key}/{k}"  # Concatenate keys with a delimiter (here '/')
                            flattened_metrics[tag] = v
                    else:
                        # Add the non-nested dictionary items
                        flattened_metrics[key] = value

                    flattened_metrics_np = {k: np.array(v) for k, v in flattened_metrics.items()}

                writer.add_scalars('Metrics', flattened_metrics_np, cur_epochs)

                print('Epoch_summary', "Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, running_loss / num_minibatch))

                # Write the loss to TensorBoard
                writer.add_scalar('Loss/train', running_loss / num_minibatch, cur_epochs)
                
                if opts.early_stopping:
                    print('cur_val_loss=', val_loss)
                    print('best_val_loss=',best_val_loss)
                    print(early_stopping_counter)
                            
                    if val_loss > best_val_loss + min_delta:
                        early_stopping_counter += 1
                        
                    elif val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stopping_counter = 0 
                    
                    else:
                        early_stopping_counter = 0 
            
            
                    if early_stopping_counter == patience - 5:
                        print('Note training may stop soon due to val loss not improving')
            
                    # Check if we need to stop training based on early stopping
                    if early_stopping_counter >= patience:
                        print(f"Stopping training early because validation loss has not improved for {patience} epochs.")
                        return

                if cur_itrs >= opts.total_itrs:
                    break

            writer.close()
            
            
    else:

        train_dst, val_dst = get_dataset(opts)
        train_loader = data.DataLoader(
            train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
        val_loader = data.DataLoader(
            val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
        print("Dataset: %s, Train set: %d, Val set: %d" %
              (opts.dataset, len(train_dst), len(val_dst)))

        # Set up model           
        if opts.model.lower()  == 'unet':
            from network.unet import UNet
            model = UNet(n_channels=3, n_classes=opts.num_classes)
            opts.backbone = 'None'
    
        elif opts.model == 'resnet-Unet':
            # from backboned_unet.backboned_unet import Unet
            model = Unet(backbone_name=[opts.backbone], pretrained=True, classes=opts.num_classes,
                         decoder_filters=(512, 256, 128, 64, 32))
    
        elif opts.model == 'efficientnet-Unet':
            # from backboned_unet.backboned_unet import Unet
            model = Unet(backbone_name=[opts.backbone], pretrained=True, classes=opts.num_classes,
                         decoder_filters=(128, 64, 32, 16, 8))
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            print('efficientNet: Number of trainable parameters:', pytorch_total_params)
            
        elif opts.model == 'pvt_v2_b-Unet':
            model = Unet(backbone_name = [opts.backbone], pretrained=True, classes=opts.num_classes,
                         decoder_filters = (128, 64, 32, 16), final_upsampling=True)
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            print('PVT: Number of trainable parameters:', pytorch_total_params)
       
        # Set up metrics
        metrics = StreamSegMetrics(opts.num_classes)
    
        # Set up optimizer              
        if opts.model == 'UNet':
            optimizer = torch.optim.Adam(params=model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
            
        elif opts.model == 'resnet-Unet' or 'efficientnet_Unet' or 'pvt_v2_b-Unet':
            optimizer = torch.optim.Adam([{'params': network.get_pretrained_parameters(), 'lr': 1e-5},
                                          {'params': network.get_random_initialized_parameters()}], lr=1e-4)
       
        else:
            optimizer = torch.optim.SGD(params=[
                {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
                {'params': model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    
        if opts.lr_policy == 'poly':
            scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
        elif opts.lr_policy == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    
        # Set up criterion
        if opts.loss_type == 'focal_loss':
            criterion = utils.FocalLoss(ignore_index=255, size_average=True)
        elif opts.loss_type == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
        print('no. of trainable hyperparameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
        def save_ckpt(path):
            """ save current model
            """
            torch.save({
                "cur_itrs": cur_itrs,
                "cur_epochs": cur_epochs,
                "best_val_loss": best_val_loss,
                "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "meanIOU_score": best_meanIOU_score,
                "Dice_score": best_dice_score,
            }, path)
            print("Model saved as %s" % path)
    
        utils.mkdir(f'checkpoints/{opts.dataset}_{opts.backbone}_lr{opts.lr}')
        
        # Restore
        best_meanIOU_score = 0.0
        best_dice_score = 0.0
        cur_itrs = 0
        best_val_loss = float('inf')
    
        if opts.ckpt is not None and os.path.isfile(os.path.join(os.getcwd(), 'checkpoints', opts.ckpt)):
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(os.path.join(os.getcwd(), 'checkpoints', opts.ckpt), map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model = model.to(device)
            if opts.continue_training:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                cur_itrs = checkpoint["cur_itrs"]
                cur_epochs = checkpoint["cur_epochs"]
                best_meanIOU_score = checkpoint['meanIOU_score']
                best_dice_score = checkpoint['Dice_score']
                best_val_loss = checkpoint['best_val_loss']
                # opts.lr = checkpoint["scheduler_state"]['_last_lr'][0]
                print("Training state restored from %s" % opts.ckpt)
            print("Model restored from %s" % opts.ckpt)
            del checkpoint  # free memory
        else:
            print("[!] Retrain")
            model = nn.DataParallel(model)
            model = model.to(device)
    
        # ==========   Train Loop   ==========#
        vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                          np.int32) if opts.enable_vis else None  # sample idxs for visualization
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images
    
        # setting up tensorboard
        writer = SummaryWriter(log_dir='logs/{}_{}_lr{}'.format(opts.dataset, opts.backbone, opts.lr))
    
        # # Initialize early stopping variables
        if opts.early_stopping:
            # stopping_threshold = 0.5  # threshold between 0 and 1, determining what percentage increase the val_loss needs to activate stopping criteria
            min_delta = 0.01
            patience = 30  # used to indicate how many iterations val_loss is above threshold before stopping
            early_stopping_counter = 0
    
        interval_loss = 0
        while True:  # cur_itrs < opts.total_itrs:
            # =====  Train  =====
            model.train()
            running_loss = 0
            num_minibatch = 0
            for (images, labels) in train_loader:
                cur_itrs += 1
                num_minibatch += 1
                cur_epochs = cur_itrs // len(train_loader)  # Update current epoch based on total iterations
    
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
  
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
                np_loss = loss.detach().cpu().numpy()
    
                running_loss += np_loss   
                interval_loss += np_loss
                if vis is not None:
                    vis.vis_scalar('Loss', cur_itrs, np_loss)
    
                if cur_itrs % 10 == 0:   
                    interval_loss = interval_loss / 10
                    print("Epoch %d, Itrs %d/%d, Loss=%f" %
                          (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
    
                    # Write the interval loss for the epoch to TensorBoard
                    writer.add_scalar('Interval_loss', interval_loss, cur_epochs, cur_itrs)
    
                    interval_loss = 0.0
    
                if cur_itrs % opts.val_interval == 0:
                    save_ckpt(f'checkpoints/{opts.dataset}_{opts.backbone}_lr{opts.lr}/latest_%s_%s_os%d_%s.pth' %
                              (opts.backbone, opts.dataset, opts.output_stride, opts.dataType))
                    print("intermediate validation...")
                    model.eval()
                    val_loss, val_score, ret_samples = validate(
                        opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                        ret_samples_ids=vis_sample_id)
                    
                    print(scheduler.get_lr())
    
                    if val_loss < best_val_loss: # saving lowest val_loss model:
                        best_val_loss = val_loss
                        save_ckpt(f'checkpoints/{opts.dataset}_{opts.backbone}_lr{opts.lr}/best_val_loss_%s_%s_os%d_%s.pth' %
                                  (opts.backbone, opts.dataset, opts.output_stride, opts.dataType))
    
                    if val_score['Mean IoU'] > best_meanIOU_score:  # save best model
                        best_meanIOU_score = val_score['Mean IoU']
                        save_ckpt(f'checkpoints/{opts.dataset}_{opts.backbone}_lr{opts.lr}/meanIOUBest_%s_%s_os%d_%s.pth' %
                                  (opts.backbone, opts.dataset, opts.output_stride, opts.dataType))
    
                    if val_score['Avg Dice Coefficient'] > best_dice_score:  # save best model
                        best_dice_score = val_score['Mean IoU']
                        save_ckpt(f'checkpoints/{opts.dataset}_{opts.backbone}_lr{opts.lr}/AvgDiceScoreBest%s_%s_os%d_%s.pth' %
                                  (opts.backbone, opts.dataset, opts.output_stride, opts.dataType))
    
                    if vis is not None:  # visualize validation score and samples
                        vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                        vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                        vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
    
                        for k, (img, target, lbl) in enumerate(ret_samples):
                            img = (denorm(img) * 255).astype(np.uint8)
                            target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                            lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                            concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                            vis.vis_image('Sample %d' % k, concat_img)
                    model.train()
                scheduler.step()
    
            save_ckpt(f'checkpoints/{opts.dataset}_{opts.backbone}_lr{opts.lr}/latest_%s_%s_os%d_%s.pth' %
                      (opts.backbone, opts.dataset, opts.output_stride, opts.dataType))
            print("validation...")
            model.eval()
            val_loss, val_score, ret_samples = validate(
                opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                ret_samples_ids=vis_sample_id)
            print(metrics.to_str(val_score))
    
            writer.add_scalar('Loss/val', val_loss, cur_epochs)
    
            # Added flattened_metrics to be able to add metrics to tensorboard
            flattened_metrics = {}
            for key, value in val_score.items():
                if isinstance(value, dict):
                    # Flatten the nested dictionary
                    for k, v in value.items():
                        tag = f"{key}/{k}"  # Concatenate keys with a delimiter (here '/')
                        flattened_metrics[tag] = v
                else:
                    # Add the non-nested dictionary items
                    flattened_metrics[key] = value
    
                flattened_metrics_np = {k: np.array(v) for k, v in flattened_metrics.items()}
    
            writer.add_scalars('Metrics', flattened_metrics_np, cur_epochs)
    
            print('Epoch_summary', "Epoch %d, Itrs %d/%d, Loss=%f" %
                  (cur_epochs, cur_itrs, opts.total_itrs, running_loss / num_minibatch))
    
            # Write the loss to TensorBoard
            writer.add_scalar('Loss/train', running_loss / num_minibatch, cur_epochs)
            
            if opts.early_stopping:
                print('cur_val_loss=', val_loss)
                print('best_val_loss=',best_val_loss)
                print(early_stopping_counter)
                        
                if val_loss > best_val_loss + min_delta:
                    early_stopping_counter += 1
                    
                elif val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0 
                
                else:
                    early_stopping_counter = 0 
        
        
                if early_stopping_counter == patience - 5:
                    print('Note training may stop soon due to val loss not improving')
        
                # Check if we need to stop training based on early stopping
                if early_stopping_counter >= patience:
                    print(f"Stopping training early because validation loss has not improved for {patience} epochs.")
                    return
    
            if cur_itrs >= opts.total_itrs:
                return
    
        writer.close()

    # tensorboard --logdir logs/efficientnet_b2_lr0.01 --port 6007

if __name__ == '__main__':
    main()