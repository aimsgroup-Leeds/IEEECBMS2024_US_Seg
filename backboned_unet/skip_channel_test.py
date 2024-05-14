def infer_skip_channels(self):

    """ Getting the number of channels at skip connections and at the output of the encoder."""

    x = torch.zeros(1, 3, 224, 224)
    has_fullres_features = self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
    channels = [] if has_fullres_features else [0]  # only VGG has features at full resolution

    # forward run in backbone to count channels (dirty solution but works for *any* Module)

    if self.backbone_name.startswith('efficientnet'):
        # Get the named children of the EfficientNet-B2 features
        named_children = list(self.backbone.features.named_children())

        # Select the desired blocks for Conv1, Conv2, Conv3, Conv4 in the UNet
        block0 = named_children[0][1]
        block1 = nn.Sequential(*[named_children[1][1], named_children[2][1]])
        block2 = named_children[3][1]
        block3 = named_children[4][1]
        block4 = nn.Sequential(*[named_children[5][1], named_children[6][1]])

        x = block0(x)
        if 'block0' in self.shortcut_features:
            channels.append(x.shape[1])

        x = block1(x)
        if 'block1' in self.shortcut_features:
            channels.append(x.shape[1])

        x = block2(x)
        if 'block2' in self.shortcut_features:
            channels.append(x.shape[1])

        x = block3(x)
        if 'block3' in self.shortcut_features:
            channels.append(x.shape[1])

        x = block4(x)
        if 'block4' in self.bb_out_name:
            #     # channels.append(x.shape[1])
            out_channels = x.shape[1]

        # out_channels = x.shape[1]


    # elif self.backbone_name.startswith('pvt_v2_b5'):
    #     # Get the named children of the EfficientNet-B2 features
    #     # named_children = list(self.backbone.stages.named_children())
    #     for name, child in self.backbone.stages.named_children():
    #         x = child(x)
    #         if name in self.shortcut_features:
    #             channels.append(x.shape[1])
    #         if name == self.bb_out_name:
    #             out_channels = x.shape[1]
    #             break

    elif self.backbone_name.startswith('pvt_v2_b5'):
        x = self.backbone.patch_embed(x)
        for name, stage in self.backbone.stages.named_children():
            x = stage(x)  # Apply each stage (block) of the PVT-tiny backbone

            # Assuming 'shortcut_features' contains the names of the stages you want to use as skip connections
            if name in self.shortcut_features:
                channels.append(x.shape[1])

            # Assuming 'bb_out_name' contains the name of the last stage (output of the backbone)
            if name == self.bb_out_name:
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
    return channels, out_channels




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
        # ... (other parts of your class)

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        self.upsample_outputs = []

        features_list = self.parallel_backbones(x)

        # Separate the feature maps from the two branches
        efficientnet_features = features_list[1:]  # Exclude the first branch (EfficientNet)
        pvt_v2_b5_features = features_list[0]  # PVT-tiny branch

        # Combine the last 4 blocks of EfficientNet with the 4 blocks of PVT-tiny
        combined_features = [
            torch.cat([ef, pt], dim=1) for ef, pt in zip(efficientnet_features, pvt_v2_b5_features)
        ]

        x = torch.cat(combined_features, dim=1)  # Concatenate the combined features

        for i, upsample_block in enumerate(self.upsample_blocks):
            x = upsample_block(x)
            self.upsample_outputs.append(x.shape)

        if self.final_upsampling:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        x = self.final_conv(x)
        return x

# ... (rest of your code)
