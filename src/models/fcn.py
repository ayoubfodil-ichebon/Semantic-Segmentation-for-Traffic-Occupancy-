"""
FCN-8s with VGG16 backbone (pre-trained on ImageNet).
"""
import torch
import torch.nn as nn
import torchvision.models as models

class FCN8s(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())

        self.enc1 = nn.Sequential(*features[0:5])   # 64
        self.enc2 = nn.Sequential(*features[5:10])  # 128
        self.enc3 = nn.Sequential(*features[10:17]) # 256
        self.enc4 = nn.Sequential(*features[17:24]) # 512
        self.enc5 = nn.Sequential(*features[24:31]) # 512

        self.classifier5 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.classifier4 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.classifier3 = nn.Conv2d(256, n_classes, kernel_size=1)

        self.deconv_up5 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, bias=False)
        self.deconv_up4 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, bias=False)
        self.deconv_up3 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=16, stride=8, bias=False)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        _, _, H, W = x.shape
        pool3 = self.enc3(self.enc2(self.enc1(x)))
        pool4 = self.enc4(pool3)
        pool5 = self.enc5(pool4)

        score5 = self.classifier5(pool5)
        score4 = self.classifier4(pool4)
        score3 = self.classifier3(pool3)

        up5 = self.deconv_up5(score5)
        fuse4 = up5 + score4
        up4 = self.deconv_up4(fuse4)
        fuse3 = up4 + score3
        out = self.deconv_up3(fuse3)

        if out.shape[2:] != (H, W):
            out = nn.functional.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
