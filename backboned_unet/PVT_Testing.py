import torch
import timm


def infer_skip_channels():

    """ Getting the number of channels at skip connections and at the output of the encoder. """

    x = torch.zeros(1, 3, 224, 224)
    channels = [0]  # only VGG has features at full resolution

    # forward run in backbone to count channels (dirty solution but works for *any* Module)

    x = backbone.patch_embed(x)
    for name, stage in backbone.stages.named_children():
        x = stage(x)  # Apply each stage (block) of the PVT-tiny backbone

        # Assuming 'shortcut_features' contains the names of the stages you want to use as skip connections
        if name in feature_names:
            channels.append(x.shape[1])

        # Assuming 'bb_out_name' contains the name of the last stage (output of the backbone)
        if name == backbone_output:
            out_channels = x.shape[1]
            break

    # else:
    #     for name, child in self.backbone.named_children():
    #         x = child(x)
    #         if name in self.shortcut_features:
    #             channels.append(x.shape[1])
    #         if name == self.bb_out_name:
    #             out_channels = x.shape[1]
    #             break

    return channels, out_channels

    
    
if __name__ == '__main__':
    
    backbone = timm.create_model('pvt_v2_b5', pretrained=True)

    # model = timm.create_model('pvt_v2_b5', pretrained=True)
    
    feature_names = ['0', '1', '2']
    backbone_output = '3'
    
    shortcut_chs, bb_out_chs = infer_skip_channels()

    # Test
    input_data = torch.randn(1, 3, 224, 224)  # Example: Batch size 1, RGB image with size 224x224
    output = backbone(input_data)

