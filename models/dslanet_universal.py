import torch
import torch.nn as nn
import torch.nn.functional as F

class SE_Block(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatCAB(nn.Module):
    def __init__(self, in_channels=16):
        super(SpatCAB, self).__init__()
        s1 = in_channels // 3
        s2 = in_channels // 3
        s3 = in_channels - s1 - s2
        self.split_sizes = [s1, s2, s3]
        
        self.conv_g1 = nn.Conv2d(s1, s1, kernel_size=3, padding=1, groups=s1, bias=False)
        self.conv_g2 = nn.Conv2d(s2, s2, kernel_size=5, padding=2, groups=s2, bias=False)
        self.conv_g3 = nn.Conv2d(s3, s3, kernel_size=7, padding=3, groups=s3, bias=False)
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        x1, x2, x3 = torch.split(x, self.split_sizes, dim=1)
        o1 = self.conv_g1(x1)
        o2 = self.conv_g2(x2)
        o3 = self.conv_g3(x3)
        out = torch.cat([o1, o2, o3], dim=1)
        return self.pointwise(out)

class SpecCAB(nn.Module):
    def __init__(self):
        super(SpecCAB, self).__init__()
        self.conv3 = nn.Conv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv5 = nn.Conv3d(1, 1, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False)
        self.conv7 = nn.Conv3d(1, 1, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False)
        self.pointwise = nn.Conv3d(3, 1, kernel_size=(1, 1, 1), bias=False)

    def forward(self, x):
        x_3d = x.unsqueeze(1)
        o1 = self.conv3(x_3d)
        o2 = self.conv5(x_3d)
        o3 = self.conv7(x_3d)
        out = torch.cat([o1, o2, o3], dim=1)
        out = self.pointwise(out)
        return out.squeeze(1)

class ABAC_Block_Universal(nn.Module):
    def __init__(self, ch_hsi=31, ch_rgb=16, use_spatcab=False, use_speccab=False):
        super(ABAC_Block_Universal, self).__init__()
        self.use_spatcab = use_spatcab
        self.use_speccab = use_speccab
        
        # RGB Branch
        self.rgb_conv1 = nn.Conv2d(ch_rgb + 1, ch_rgb, kernel_size=3, padding=1, padding_mode='reflect')
        self.rgb_act = nn.LeakyReLU(0.2, inplace=True)
        self.rgb_conv2 = nn.Conv2d(ch_rgb, ch_rgb, kernel_size=3, padding=1, padding_mode='reflect')
        
        if self.use_spatcab:
            self.spatial_scab = SpatCAB(ch_rgb)
            
        # Bridge
        self.gamma_conv = nn.Conv2d(ch_rgb, ch_hsi, kernel_size=3, padding=1)
        self.beta_conv  = nn.Conv2d(ch_rgb, ch_hsi, kernel_size=3, padding=1)
        
        # HSI Branch
        self.hsi_depthwise = nn.Conv2d(ch_hsi, ch_hsi, kernel_size=3, padding=1, groups=ch_hsi, bias=False)
        self.hsi_bn = nn.BatchNorm2d(ch_hsi)
        self.hsi_pointwise = nn.Conv2d(ch_hsi, ch_hsi, kernel_size=1, bias=False)
        self.hsi_act = nn.ReLU(inplace=True)
        self.se = SE_Block(ch_hsi, reduction=4)
        
        if self.use_speccab:
            self.spectral_scab = SpecCAB()

    def forward(self, x_hsi, x_rgb):
        res_hsi = x_hsi
        res_rgb = x_rgb
        
        uncertainty = torch.std(x_hsi, dim=1, keepdim=True)
        
        rgb_fused = torch.cat([x_rgb, uncertainty], dim=1)
        out_rgb = self.rgb_act(self.rgb_conv1(rgb_fused))
        out_rgb = self.rgb_conv2(out_rgb)
        
        gamma = self.gamma_conv(out_rgb)
        beta  = self.beta_conv(out_rgb)
        
        out_hsi = self.hsi_depthwise(x_hsi)
        out_hsi = self.hsi_bn(out_hsi)
        out_hsi = out_hsi * (1 + gamma) + beta
        out_hsi = self.hsi_act(out_hsi)
        out_hsi = self.hsi_pointwise(out_hsi)
        out_hsi = self.se(out_hsi)
        
        out_hsi_final = out_hsi + res_hsi
        if self.use_speccab:
            out_hsi_final = self.spectral_scab(out_hsi_final)
            
        out_rgb_final = out_rgb + res_rgb
        if self.use_spatcab:
            out_rgb_final = self.spatial_scab(out_rgb_final)
            
        return out_hsi_final, out_rgb_final

class DSLAnet_Universal(nn.Module):
    def __init__(self, num_bands=31, num_rgb_features=16, num_blocks=8, use_spatcab=False, use_speccab=False):
        super(DSLAnet_Universal, self).__init__()
        self.head_hsi = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1, padding_mode='reflect')
        self.head_rgb = nn.Conv2d(3, num_rgb_features, kernel_size=3, padding=1, padding_mode='reflect')
        
        self.blocks = nn.ModuleList([
            ABAC_Block_Universal(num_bands, num_rgb_features, use_spatcab, use_speccab)
            for _ in range(num_blocks)
        ])
        
        # Original tail was SpatCAB on 31 channels.
        # Note: The original DSLAnet used a SpatCAB (formerly Micro_SCAB) tail on the HSI output.
        # To strictly maintain equivalence to the original ablation strategy, we keep this baseline tail.
        self.tail_hsi = SpatCAB(num_bands)

    def forward(self, cassi_shiftback, rgb_real):
        x_hsi = self.head_hsi(cassi_shiftback)
        x_rgb = self.head_rgb(rgb_real)
        
        for block in self.blocks:
            x_hsi, x_rgb = block(x_hsi, x_rgb)
            
        out = self.tail_hsi(x_hsi)
        return out
