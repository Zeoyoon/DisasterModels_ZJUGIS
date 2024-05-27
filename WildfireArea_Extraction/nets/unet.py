import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
from nets.attention import *
attention_block =[se_block,cbam_block,eca_block]
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs




# phi = 0 : 不添加注意力机制
# phi = 1 : se
# phi = 2 : cbam
# phi = 3 : eca

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg',phi=2):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        self.phi = phi
        if phi >= 1 and phi <= 3: #注意力机制的设置
            self.feat4_attention = attention_block[phi - 1](512)
            self.feat3_attention = attention_block[phi - 1](256)
            self.feat2_attention = attention_block[phi - 1](128)
            self.feat1_attention = attention_block[phi - 1](64)
        if phi >= 1 and phi <= 3:  # 注意力机制的设置
            self.up_concat4_attention = attention_block[phi - 1](512)
            self.up_concat3_attention = attention_block[phi - 1](256)
            self.up_concat2_attention = attention_block[phi - 1](128)
            self.up_concat1_attention = attention_block[phi - 1](64)
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)


        up4 = self.up_concat4(feat4, feat5)
        if self.phi >= 1 and self.phi <= 3:  # 注意力机制的堆叠  3\4
            up4 = self.up_concat4_attention (up4)
        up3 = self.up_concat3(feat3, up4)
        if self.phi >= 1 and self.phi <= 3:
            up3 = self.up_concat3_attention (up3)
        up2 = self.up_concat2(feat2, up3)
        if self.phi >= 1 and self.phi <= 3:
            up2 = self.up_concat2_attention(up2)
        up1 = self.up_concat1(feat1, up2)
        if self.phi >= 1 and self.phi <= 3:
            up1 = self.up_concat1_attention(up1)
        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
