"""
Created on Thu Aug 10 14:40:27 2023

PBA Testing

"""
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.nn import functional as F
import timm

def get_backbone(name, pretrained=True):
    """ Loading backbone, defining names for skip-connections and encoder output. """

    # TODO: More backbones

    # loading backbone model
    if name == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        backbone = models.resnet152(pretrained=pretrained)
    elif name == 'vgg16':
        backbone = models.vgg16_bn(pretrained=pretrained).features
    elif name == 'vgg19':
        backbone = models.vgg19_bn(pretrained=pretrained).features
        # Added efficientNet 
    elif name == 'efficientnet_b0':
        backbone = models.efficientnet_b0(pretrained=pretrained)
    elif name == 'efficientnet_b1':
        backbone = models.efficientnet_b1(pretrained=pretrained)
    elif name == 'efficientnet_b2':
        backbone = models.efficientnet_b2(pretrained=pretrained)
    elif name == 'efficientnet_b3':
        backbone = models.efficientnet_b3(pretrained=pretrained)
    elif name == 'efficientnet_b4':
        backbone = models.efficientnet_b4(pretrained=pretrained)
    elif name == 'efficientnet_b5':
        backbone = models.efficientnet_b5(pretrained=pretrained)
    elif name == 'efficientnet_b6':
        backbone = models.efficientnet_b6(pretrained=pretrained)
    elif name == 'efficientnet_b7':
        backbone = models.efficientnet_b7(pretrained=pretrained)

    elif name == ('pvt_v2_b5'):
        backbone = timm.create_model('pvt_v2_b5', pretrained=pretrained)

    # End of added section
    # elif name == 'inception_v3':
    #     backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
    elif name == 'densenet121':
        backbone = models.densenet121(pretrained=True).features
    elif name == 'densenet161':
        backbone = models.densenet161(pretrained=True).features
    elif name == 'densenet169':
        backbone = models.densenet169(pretrained=True).features
    elif name == 'densenet201':
        backbone = models.densenet201(pretrained=True).features
    elif name == 'unet_encoder':
        from unet_backbone import UnetEncoder
        backbone = UnetEncoder(3)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    elif name == 'vgg16':
        # TODO: consider using a 'bridge' for VGG models, there is just a MaxPool between last skip and backbone output
        feature_names = ['5', '12', '22', '32', '42']
        backbone_output = '43'
    elif name == 'vgg19':
        feature_names = ['5', '12', '25', '38', '51']
        backbone_output = '52'
    # elif name == 'inception_v3':
    #     feature_names = [None, 'Mixed_5d', 'Mixed_6e']
    #     backbone_output = 'Mixed_7c'
    elif name.startswith('densenet'):
        feature_names = [None, 'relu0', 'denseblock1', 'denseblock2', 'denseblock3']
        backbone_output = 'denseblock4'
    elif name == 'unet_encoder':
        feature_names = ['module1', 'module2', 'module3', 'module4']
        backbone_output = 'module5'
    elif name.startswith('efficientnet'):
        feature_names = [None, 'block0', 'block1', 'block2', 'block3']  # efficientnet features: 0 ->
        # block0, 1,2 -> block1, 3->block2, 4->block3, 5,6 -> block4
        backbone_output = 'block4'
    elif name.startswith('pvt_v2_b5'):
        feature_names = [None, '0', '1', '2']
        backbone_output = '3'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone, feature_names, backbone_output

class UpsampleBlock(nn.Module):

    # TODO: separate parametric and non-parametric classes?
    # TODO: skip connection concatenated OR added

    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, parametric=False, skip_agg = 'concat'):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        self.skip_agg = skip_agg
        ch_out = ch_in / 2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

    def forward(self, x, skip_connection=None):

        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            if self.skip_agg == 'concat':
                x = torch.cat([x, skip_connection], dim=1)
            elif self.skip_agg == 'addition':
                x = x + skip_connection
            else:
                raise ValueError("Invalid value for skip_agg. Expected 'concat' or 'addition', but got: {}".format(self.skip_agg))

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x


class ParallelBackbones(nn.Module):
    def __init__(self, backbone_names, pretrained=True):
        super(ParallelBackbones, self).__init__()
        self.backbone_names = backbone_names
        self.backbones = nn.ModuleList([get_backbone(name, pretrained)[0] for name in backbone_names])
        
    def forward_backbone(self, x):
        features_list = []
        feature_outputs  = []
    
        for backbone_name in self.backbone_names:
            
            features = {None: None}
            x = torch.zeros(1, 3, 224, 224)
            
            if backbone_name.startswith('efficientnet'):
                
                backbone, shortcut_features, bb_out_name = get_backbone(backbone_name, pretrained=True)
                
                named_children = list(backbone.features.named_children())
                block0 = named_children[0][1]
                block1 = nn.Sequential(*[named_children[1][1], named_children[2][1]])
                block2 = named_children[3][1]
                block3 = named_children[4][1]
                block4 = nn.Sequential(*[named_children[5][1], named_children[6][1]])
    
                x = block0(x)
                features['block0'] = x
                
    
                x = block1(x)
                features['block1'] = x
    
                x = block2(x)
                features['block2'] = x
    
                x = block3(x)
                features['block3'] = x
    
                x = block4(x)
                # features['block4'] = x
    
            elif backbone_name.startswith('pvt_v2_b5'):
                
                backbone, shortcut_features, bb_out_name = get_backbone(backbone_name, pretrained=True)
                
                x = backbone.patch_embed(x)
                for name, stage in backbone.stages.named_children():
                    x = stage(x)  # Apply each stage (block) of the PVT-tiny backbone
    
                    if name in shortcut_features:
                        features[name] = x
    
                    if name == bb_out_name:
                        break
    
            else:
                backbone, shortcut_features, bb_out_name = get_backbone(backbone_name, pretrained=True)

                for name, child in backbone.named_children():
                    x = child(x)
                    if name in shortcut_features:
                        features[name] = x
                    if name == bb_out_name:
                        break
    
            features_list.append(features)
            feature_outputs.append(x)
    
        return feature_outputs, features_list


    def forward(self, x):
        # outputs = []
        # for backbone in self.backbones:
        # features = self.forward_backbone(x)
        # outputs.append(features)
        outputs = self.forward_backbone(x)
        return outputs
    
class MultiBackboneUnet(nn.Module):
    def __init__(
        self,
        backbone_names=["pvt_v2_b5", "efficientnet_b2"],
        pretrained=True,
        encoder_freeze=False,
        classes=2,
        decoder_filters=(256, 128, 64, 32, 16),
        parametric_upsampling=True,
        shortcut_features="default",
        decoder_use_batchnorm=True,
        final_upsampling=False,
    ):
        super(MultiBackboneUnet, self).__init__()
        
        self.backbone_names = backbone_names
        
        shortcut_feature_lst_lengths = []
        for backbone_name in self.backbone_names:
            backbone, shortcut_features, bb_out_name = get_backbone(backbone_name, pretrained=True)
            shortcut_feature_lst_lengths.append(len(shortcut_features))
            

        self.final_upsampling = final_upsampling

        self.parallel_backbones = ParallelBackbones(backbone_names, pretrained)

        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if shortcut_features != "default":
            self.shortcut_features = shortcut_features

        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[: max(shortcut_feature_lst_lengths)]
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            self.upsample_blocks.append(
                UpsampleBlock(
                    filters_in,
                    filters_out,
                    skip_in=shortcut_chs[num_blocks - i - 1],
                    parametric=parametric_upsampling,
                    use_bn=decoder_use_batchnorm,
                    skip_agg='concat',
                )
            )

        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()

        self.replaced_conv1 = False  # for accommodating inputs with different numbers of channels later
        
    def infer_skip_channels(self):
        
        shortcut_chs_list = []
        bb_out_chs_list = []
        for backbone_name in self.backbone_names:
            x = torch.zeros(1, 3, 224, 224)
            channels = [0]
            if backbone_name.startswith('efficientnet'):
                
                backbone, shortcut_features, bb_out_name = get_backbone(backbone_name, pretrained=True)
                
                # Get the named children of the EfficientNet-B2 features
                named_children = list(backbone.features.named_children())

                # Select the desired blocks for Conv1, Conv2, Conv3, Conv4 in the UNet
                block0 = named_children[0][1]
                block1 = nn.Sequential(*[named_children[1][1], named_children[2][1]])
                block2 = named_children[3][1]
                block3 = named_children[4][1]
                block4 = nn.Sequential(*[named_children[5][1], named_children[6][1]])

                x = block0(x)
                if 'block0' in shortcut_features:
                    channels.append(x.shape[1])

                x = block1(x)
                if 'block1' in shortcut_features:
                    channels.append(x.shape[1])

                x = block2(x)
                if 'block2' in shortcut_features:
                    channels.append(x.shape[1])

                x = block3(x)
                if 'block3' in shortcut_features:
                    channels.append(x.shape[1])

                x = block4(x)
                if 'block4' in bb_out_name:
                    #     # channels.append(x.shape[1])
                    out_channels = x.shape[1]
            
            elif backbone_name.startswith('pvt_v2_b5'):
                
                backbone, shortcut_features, bb_out_name = get_backbone(backbone_name, pretrained=True)
                
                x = backbone.patch_embed(x)
                for name, stage in backbone.stages.named_children():
                    x = stage(x)  # Apply each stage (block) of the PVT-tiny backbone
        
                    # Assuming 'shortcut_features' contains the names of the stages you want to use as skip connections
                    if name in shortcut_features:
                        channels.append(x.shape[1])
        
                    # Assuming 'bb_out_name' contains the name of the last stage (output of the backbone)
                    if name == bb_out_name:
                        out_channels = x.shape[1]
                        break
                    
            else:
                for name, child in self.backbone.named_children():
                    x = child(x)
                    if name in self.shortcut_features:
                        channels.append(x.shape[1])
                    if name == self.bb_out_name:
                        out_channels = x.shape[1]
                        break

                        
            shortcut_chs_list.append(channels)
            bb_out_chs_list.append(out_channels)
            
        if len(shortcut_chs_list[0]) >= len(shortcut_chs_list[1]):
            longer_list = shortcut_chs_list[0]
            shorter_list = shortcut_chs_list[1]
        else:
            longer_list = shortcut_chs_list[1]
            shorter_list = shortcut_chs_list[0]
            
        num_elements_to_keep = len(longer_list) - len(shorter_list)

        # Sum corresponding elements using zip and list comprehension
        combined_shortcut_chs = longer_list[:num_elements_to_keep] + [x + y for x, y in zip(longer_list[num_elements_to_keep:], shorter_list)]
        
        
        # combined_shortcut_chs = [sum(chs) for chs in zip(*shortcut_chs_list)]
        combined_bb_out_chs = sum(bb_out_chs_list)

        return combined_shortcut_chs, combined_bb_out_chs

    # def forward(self, *inputs):
    #     x = torch.cat(inputs, dim=1)
    #     self.upsample_outputs = []
    
    #     feature_outputs, features_list = self.parallel_backbones(x)
    
    #     # Determine the minimum spatial dimensions among the skip features
    #     min_height = min([features[skip_name].shape[2] for features in features_list for skip_name in features.keys() if features[skip_name] is not None])
    #     min_width = min([features[skip_name].shape[3] for features in features_list for skip_name in features.keys() if features[skip_name] is not None])
    
    #     skip_features_list = []
    #     for features in features_list:
    #         resized_skip_features = {}
    #         for skip_name in features.keys():
    #             if features[skip_name] is not None:
    #                 skip_feature = features[skip_name]
    #                 resized_skip_feature = F.interpolate(skip_feature, size=(min_height, min_width), mode='bilinear', align_corners=False)
    #                 resized_skip_features[skip_name] = resized_skip_feature
        
    #         # Sum the resized skip features using element-wise addition
    #         skip_features = torch.cat(list(resized_skip_features.values()), dim=1)
    #         skip_features_list.append(skip_features)
    
    #     x = sum(skip_features_list)
    
    #     for i, upsample_block in enumerate(self.upsample_blocks):
    #         x = upsample_block(x)
    #         self.upsample_outputs.append(x.shape)
    
    #     if self.final_upsampling:
    #         x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
    
    #     x = self.final_conv(x)
    #     return x
    

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        self.upsample_outputs = []
        skip_features = {}

        feature_outputs, features_list = self.parallel_backbones(x)
        
        adjusted_skip_features = {}
        
        for feature in features_list:
            for skip_name in feature.keys():
                skip_features[skip_name] = feature[skip_name]
                
        
        # skip_features['0'] -> skip_features['block1']
        # skip_features['1'] -> skip_features['block2']
        # skip_features['2'] -> skip_features['block3']
        
        # Step 2 & 3: Apply 1x1 convolution and sum specific pairs of skip features
        if '0' in skip_features and 'block1' in skip_features:
            # Apply 1x1 convolution to adjust channel dimensions for '0'
            adjusted_0 = nn.Conv2d(in_channels=skip_features['0'].shape[1], out_channels=64, kernel_size=1)
            
            # Apply 1x1 convolution to adjust channel dimensions for 'block1'
            adjusted_block1 = nn.Conv2d(in_channels=skip_features['block1'].shape[1], out_channels=64, kernel_size=1)
            
            # Sum the adjusted skip features using element-wise addition
            adjusted_skip_features['0_block1'] = adjusted_0(skip_features['0']) + adjusted_block1(skip_features['block1'])
        else:
            adjusted_skip_features['0_block1'] = None
        
        if '1' in skip_features and 'block2' in skip_features:
            # Apply 1x1 convolution to adjust channel dimensions for '1'
            adjusted_1 = nn.Conv2d(in_channels=skip_features['1'].shape[1], out_channels=128, kernel_size=1)
            
            # Apply 1x1 convolution to adjust channel dimensions for 'block2'
            adjusted_block2 = nn.Conv2d(in_channels=skip_features['block2'].shape[1], out_channels=128, kernel_size=1)
            
            # Sum the adjusted skip features using element-wise addition
            adjusted_skip_features['1_block2'] = adjusted_1(skip_features['1']) + adjusted_block2(skip_features['block2'])
        else:
            adjusted_skip_features['1_block2'] = None
        
        if '2' in skip_features and 'block3' in skip_features:
            # Apply 1x1 convolution to adjust channel dimensions for '2'
            adjusted_2 = nn.Conv2d(in_channels=skip_features['2'].shape[1], out_channels=320, kernel_size=1)
            
            # Apply 1x1 convolution to adjust channel dimensions for 'block3'
            adjusted_block3 = nn.Conv2d(in_channels=skip_features['block3'].shape[1], out_channels=320, kernel_size=1)
            
            # Sum the adjusted skip features using element-wise addition
            adjusted_skip_features['2_block3'] = adjusted_2(skip_features['2']) + adjusted_block3(skip_features['block3'])
        else:
            adjusted_skip_features['2_block3'] = None
        
            
        
        # Sum the intermediate outputs from each parallel encoder
        summed_features = torch.stack(feature_outputs, dim=0).sum(dim=0)
        # summed_features = torch.sum(summed_features, dim=0)

        for feature in features_list:
            for skip_name, upsample_block in zip(list(reversed(feature.keys())), self.upsample_blocks):
                skip_features = {skip_name: feature[skip_name]}
                x = upsample_block(summed_features, skip_features[skip_name])
                self.upsample_outputs.append(x.shape)

        if self.final_upsampling:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.final_conv(x)
        return x

    

if __name__ == '__main__':

    net = MultiBackboneUnet(backbone_names=["pvt_v2_b5", "efficientnet_b2"], decoder_filters=[128, 64, 32, 16, 8])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    print('Network initialized. Running a test batch.')
    for _ in range(1):
        with torch.set_grad_enabled(True):
            batch = torch.empty(1, 3, 384, 384).normal_()
            targets = torch.empty(1, 2, 384, 384).normal_()

            out = net(batch)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        print(out.shape)

    print('Done.')
