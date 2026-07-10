import torch
import torch.nn as nn

class CPEN(nn.Module):
    def __init__(self, n_feats=64, n_encoder_res=6):
        super(CPEN, self).__init__()

        self.pixel_unshuffle = nn.PixelUnshuffle(4)

        self.downsample_layers = nn.ModuleList([
            # 第一层: [B, 3072, 32, 32] -> [B, 96, 128, 128]
            nn.Sequential(
                nn.Conv2d(3072, 768, kernel_size=1),  # 降通道
                nn.Conv2d(768, 96, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)  # 上采样到128x128
            ),
            # 第二层: [B, 3072, 32, 32] -> [B, 192, 64, 64]
            nn.Sequential(
                nn.Conv2d(3072, 768, kernel_size=1),
                nn.Conv2d(768, 192, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)  # 上采样到64x64
            ),
            # 第三层: [B, 3072, 32, 32] -> [B, 384, 32, 32]
            nn.Sequential(
                nn.Conv2d(3072, 768, kernel_size=1),
                nn.Conv2d(768, 384, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True)
            ),
            # 第四层: [B, 3072, 32, 32] -> [B, 768, 16, 16]
            nn.Sequential(
                nn.Conv2d(3072, 768, kernel_size=1),
                nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.1, True)
            )
        ])

    def forward(self, x, gt):
        # 存储中间特征
        feature_list = []

        if x.dim() == 4 and x.shape[-1] == 96:
            x = x.permute(0, 3, 1, 2)
            gt = gt.permute(0, 3, 1, 2)

        # 应用 pixel_unshuffle
        gt0 = self.pixel_unshuffle(gt)  # [B, 1536, 32, 32]
        x0 = self.pixel_unshuffle(x)  # [B, 1536, 32, 32]

        x_cat = torch.cat([x0, gt0], dim=1)  # [B, 3072, 32, 32]

        for i, layer in enumerate(self.downsample_layers):
            current_feat = layer(x_cat)

            feat_output = current_feat.permute(0, 2, 3, 1)
            feature_list.append(feat_output)

        return feature_list