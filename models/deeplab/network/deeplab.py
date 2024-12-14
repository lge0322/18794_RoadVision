import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        self.aspp_conv = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=dilation,
                                   dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.aspp_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()

        self.aspp_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
      
        input_size = x.shape[-2:]
        x = self.aspp_pooling(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = F.interpolate(x, 
                          size=input_size, 
                          mode="bilinear", 
                          align_corners=False)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()

        # Initializing modules
        modules = []

        # (a) One 1x1 convolution 
        self.num_filters = 256
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels,
                             out_channels=self.num_filters,
                             kernel_size=1)
        self.bn_1 = nn.BatchNorm2d(self.num_filters)
        self.relu_1 = nn.ReLU()

        modules.append(
            nn.Sequential(self.conv_1x1, self.bn_1, self.relu_1)
        )
        
        # (b) Three 3x3 convolutions with the different atrous_rates
        # Should be (6, 12, 18) as mentioned in the paper
        for atrous_rate in atrous_rates:
            # Perform the ASPP Convolution with the atrous rate (dilation)
            aspp_conv = ASPPConv(in_channels=in_channels,
                                 out_channels=self.num_filters,
                                 dilation=atrous_rate)
            modules.append(aspp_conv)

        # Global Average Pooling
        self.aspp_pooling = ASPPPooling(in_channels=in_channels,
                                        out_channels=self.num_filters)
        modules.append(self.aspp_pooling)

        # Hold submodules in a list
        self.submodules = nn.ModuleList(modules)

        # 1x1 conv after the concatenation
        concat_inchannel_size = len(self.submodules) * self.num_filters

        self.concat_conv_1x1 = nn.Conv2d(in_channels=concat_inchannel_size, 
                                         out_channels=self.num_filters, 
                                         kernel_size=1)
        self.concat_bn = nn.BatchNorm2d(self.num_filters)
        self.concat_relu = nn.ReLU()

        self.concat_conv = nn.Sequential(
            self.concat_conv_1x1,
            self.concat_bn,
            self.concat_relu
        )

    def forward(self, x):
      
        # Convert x is to a tensor here
        features = []
        for submodule in self.submodules:
            feature = submodule(x)
            features.append(feature)

        # Concatenation of features
        x = torch.cat(features, dim=1)

        # Apply the 1x1 conv layer to the concatenated features
        x = self.concat_conv(x)
        return x


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()
    
        # The model should have the following 3 arguments
        #   in_channels: number of input channels
        #   num_classes: number of classes for prediction
        #   aspp_dilate: atrous_rates for ASPP
        #
        # ================================================================================ #
        self._init_weight()
        self.num_filters = 256
        self.aspp = ASPP(in_channels=in_channels,
                         atrous_rates=aspp_dilate)
        self.output_layer = nn.Conv2d(self.num_filters, num_classes, kernel_size=1)


    def forward(self, feature):
        feature = feature['out']
        x = self.aspp(feature)
        output = self.output_layer(x)
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        # The model should have the following 4 arguments
        #   in_channels: number of input channels
        #   low_level_channels: number of channels for project
        #   num_classes: number of classes for prediction
        #   aspp_dilate: atrous_rates for ASPP
        #
        # ================================================================================ #
        self._init_weight()
        self.num_filters = 256
        # Encoder module
        self.encoder = ASPP(in_channels=in_channels,
                            atrous_rates=aspp_dilate)

        # Decoder module
        self.conv_1x1_dec = nn.Conv2d(in_channels=low_level_channels,
                             out_channels=self.num_filters,
                             kernel_size=1)
        self.bn_1_dec = nn.BatchNorm2d(self.num_filters)
        self.relu_1_dec = nn.ReLU()

        self.decoder = nn.Sequential(
            self.conv_1x1_dec,
            self.bn_1_dec,
            self.relu_1_dec
        )

        # Convolution 3x3 layer
        concat_size = 2 * low_level_channels
        self.conv_3x3 = nn.Conv2d(in_channels=concat_size,
                             out_channels=self.num_filters,
                             kernel_size=3)
        self.bn_2 = nn.BatchNorm2d(self.num_filters)
        self.relu_2 = nn.ReLU()

        # Output layer
        self.output_layer = nn.Conv2d(self.num_filters, num_classes, kernel_size=1)
        

    def forward(self, feature):
        # Put the features through the encoder
        encoder_feature = feature['out']
        input_shape = encoder_feature.shape[-2:]

        encoder_x = self.encoder(encoder_feature)

        # Upsampling the output of the encoder
        decoder_feature = feature['low_level']
        low_level_feature_shape = decoder_feature.shape[-2:]

        encoder_x = F.interpolate(encoder_x, 
                          size=low_level_feature_shape, 
                          mode="bilinear", 
                          align_corners=False)

        decoder_x = self.decoder(decoder_feature)

        # Concatenate the encoder and decoder outputs
        concat_x = torch.cat([encoder_x, decoder_x], dim=1)
        
        # 3x3 convolution layer
        x = self.conv_3x3(concat_x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        # Upsample the final image by 4
        upsampler = nn.UpsamplingBilinear2d(scale_factor=4)
        x = upsampler(x)

        # Output layer
        output = self.output_layer(x)
        
        return output
    

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)