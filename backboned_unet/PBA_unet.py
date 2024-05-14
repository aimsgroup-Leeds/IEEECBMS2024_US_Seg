import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.nn import functional as F
import timm


class UNet(nn.Module):
    def __init__(self, num_classes, pvt_backbone='pvt_v2_b5', efficient_backbone='efficientnet_b2'):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        # Load the PVT-Tiny backbone
        self.pvt_backbone = timm.create_model(pvt_backbone, pretrained=True)
        self.pvt_encoder_channels = [64, 128, 320, 512]  # Adjust these based on the model architecture

        # Load the EfficientNet backbone
        self.efficient_backbone = timm.create_model(efficient_backbone, pretrained=True)
        self.efficient_encoder_channels = [24, 40, 112, 320]  # Adjust these based on the model architecture

        # Define decoder blocks for skip connections
        self.decoder1 = self._make_decoder_block(self.pvt_encoder_channels[0] + self.efficient_encoder_channels[0], 64)
        self.decoder2 = self._make_decoder_block(self.pvt_encoder_channels[1] + self.efficient_encoder_channels[1], 128)
        self.decoder3 = self._make_decoder_block(self.pvt_encoder_channels[2] + self.efficient_encoder_channels[2], 320)
        self.decoder4 = self._make_decoder_block(self.pvt_encoder_channels[3] + self.efficient_encoder_channels[3], 512)

        # Final convolutional layer to get the desired number of output channels
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Forward pass through the PVT-Tiny backbone
        pvt_features = self.pvt_backbone.forward_features(x)
        pvt_x1 = pvt_features[0]
        pvt_x2 = pvt_features[1]
        pvt_x3 = pvt_features[2]
        pvt_x4 = pvt_features[3]

        # Forward pass through the EfficientNet backbone
        efficient_features = self.efficient_backbone.extract_features(x)
        efficient_x1 = efficient_features[0]
        efficient_x2 = efficient_features[1]
        efficient_x3 = efficient_features[2]
        efficient_x4 = efficient_features[3]

        # Combine features with skip connections and decoder blocks
        x = self.decoder1(torch.cat([pvt_x1, efficient_x1], dim=1))
        x = self.decoder2(torch.cat([pvt_x2, efficient_x2], dim=1))
        x = self.decoder3(torch.cat([pvt_x3, efficient_x3], dim=1))
        x = self.decoder4(torch.cat([pvt_x4, efficient_x4], dim=1))

        # Apply final convolution to get the segmentation mask
        x = self.final_conv(x)

        return x

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
