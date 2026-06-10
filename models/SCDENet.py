import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

__all__ = ["DRPCANet"]


class StripPoolingBlock(nn.Module):
    def __init__(self, channel):
        super(StripPoolingBlock, self).__init__()

        self.conv_local = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(True)
        )

        # 水平条纹路径 (1 x W)
        self.path_hor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channel, channel // 2, 1, 1, 0),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(True),
        )

        # 垂直条纹路径 (H x 1)
        self.path_ver = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channel, channel // 2, 1, 1, 0),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(True),
        )

        # 融合层
        self.fusion = nn.Conv2d(channel + channel, channel, 1, 1)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # 1. 局部特征
        local_feat = self.conv_local(x)

        # 2. 条纹上下文 (Strip Context)
        feat_hor = self.path_hor(x)
        feat_hor = F.interpolate(feat_hor, (h, w), mode="bilinear", align_corners=False)

        feat_ver = self.path_ver(x)
        feat_ver = F.interpolate(feat_ver, (h, w), mode="bilinear", align_corners=False)

        # 3. 融合 (Local + Horizontal + Vertical)
        # 这样能把横向和纵向的背景杂波都"减掉"
        context = torch.cat([feat_hor, feat_ver], dim=1)  # [B, C, H, W]
        cat = torch.cat([local_feat, context], dim=1)  # [B, 2C, H, W]

        out = self.fusion(cat)

        # 残差连接
        return self.relu(out + x)


# # --- 变体: GAP-Block (用于消融实验对比) ---
# # 目的：证明 Strip Pooling 比普通的 Global Average Pooling (GAP) 更好
# # 结构：Local (3x3) + Global (GAP)
# class StripPoolingBlock(nn.Module):
#     def __init__(self, channel):
#         super(StripPoolingBlock, self).__init__()

#         # 1. 局部路径 (保持不变)
#         self.conv_local = nn.Sequential(
#             nn.Conv2d(channel, channel, 3, 1, 1),
#             nn.BatchNorm2d(channel), nn.ReLU(True)
#         )

#         # 2. 全局路径 (改为普通的全局平均池化)
#         # 相比 Strip Pooling，GAP 会丢失空间方向信息
#         self.global_path = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), # GAP: [B, C, H, W] -> [B, C, 1, 1]
#             nn.Conv2d(channel, channel, 1, 1, 0),
#             nn.BatchNorm2d(channel), nn.ReLU(True)
#         )

#         # 3. 融合层 (输入是 2*channel)
#         self.fusion = nn.Conv2d(channel + channel, channel, 1, 1)
#         self.relu = nn.ReLU(True)

#     def forward(self, x):
#         h, w = x.shape[2], x.shape[3]

#         # Local
#         local_feat = self.conv_local(x)

#         # Global (GAP)
#         # 需要把 1x1 的特征上采样回 HxW
#         global_feat = self.global_path(x)
#         global_feat = F.interpolate(global_feat, (h, w), mode='nearest')

#         # Concat
#         cat = torch.cat([local_feat, global_feat], dim=1)

#         out = self.fusion(cat)
#         return self.relu(out + x)


class LowrankModule(nn.Module):
    def __init__(self, channel=32, layers=3):
        super(LowrankModule, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(1, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(True)
        )

        # 使用条纹池化块
        self.body = nn.Sequential(
            StripPoolingBlock(channel),
            StripPoolingBlock(channel),
            StripPoolingBlock(channel),
        )

        self.tail = nn.Conv2d(channel, 1, 3, 1, 1)

    def forward(self, D, T):
        x = D - T
        feat = self.head(x)
        feat = self.body(feat)
        B = x + self.tail(feat)
        return B


class parametergenerator(nn.Module):
    def __init__(self, midchannel=3):
        super(parametergenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，提取全局特征
            nn.Conv2d(1, midchannel, kernel_size=1, bias=True),  # 卷积升维
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(midchannel, 1, kernel_size=1, bias=True),  # 卷积恢复维度
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.generator(x)


# --- 其他类保持不变 (DRPCANet, DynamicSparseModule 等) ---
class DRPCANet(nn.Module):
    def __init__(
        self, stage_num=6, slayers=6, llayers=3, mlayers=5, channel=32, mode="train"
    ):
        super(DRPCANet, self).__init__()
        self.stage_num = stage_num
        self.decos = nn.ModuleList()
        self.mode = mode
        for _ in range(stage_num):
            self.decos.append(
                DecompositionModule(
                    slayers=slayers, llayers=llayers, mlayers=mlayers, channel=channel
                )
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, D):
        T = torch.zeros(D.shape).to(D.device)
        for i in range(self.stage_num):
            D, T = self.decos[i](D, T)
        if self.mode == "train":
            return D, T
        else:
            return T


class DecompositionModule(nn.Module):
    def __init__(self, slayers=6, llayers=3, mlayers=5, channel=32):
        super(DecompositionModule, self).__init__()
        self.lowrank = LowrankModule(channel=channel, layers=llayers)
        self.sparse = DynamicSparseModule(channel=channel, layers=slayers)
        self.merge = DynamicResidualMergeModule(channel=channel, layers=mlayers)

    def forward(self, D, T):
        B = self.lowrank(D, T)
        T = self.sparse(D, B, T)
        D = self.merge(B, T)
        return D, T


class DynamicSparseModule(nn.Module):
    def __init__(self, channel=32, layers=6):
        super(DynamicSparseModule, self).__init__()
        convs = [
            nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
        ]
        for i in range(layers):
            # --- 修改点: 使用膨胀卷积 (dilation=2) ---
            # padding=2 是为了保持尺寸不变 (K=3, D=2 -> P=2)
            convs.append(
                nn.Conv2d(
                    channel, channel, kernel_size=3, padding=2, dilation=2, stride=1
                )
            )
            convs.append(nn.ReLU(True))

        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)

        # 参数生成器保持 Baseline 原版 (AvgPool)
        self.gamma_generator = parametergenerator(midchannel=3)
        self.epsilon_generator = parametergenerator(midchannel=3)

    def forward(self, D, B, T):
        gamma = self.gamma_generator(T)
        x = gamma * T + (1 - gamma) * (D - B)
        epsilon = self.epsilon_generator(x)
        T = x - epsilon * self.convs(x)
        return T


class DynamicResidualMergeModule(nn.Module):
    def __init__(self, channel=32, layers=5):
        super(DynamicResidualMergeModule, self).__init__()

        convs = [
            nn.Conv2d(2, channel, kernel_size=1),  # 这里的 2 是关键
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]

        convs.append(
            DynamicResidualGroup(
                default_conv, channel, kernel_size=3, reduction=16, n_resblocks=layers
            )
        )
        convs.append(nn.Conv2d(channel, 1, kernel_size=1))
        self.mapping = nn.Sequential(*convs)

    def forward(self, B, T):
        x = torch.cat([B, T], dim=1)  # [Batch, 2, H, W] <-- 新代码
        D = self.mapping(x)
        return D


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=32, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class DynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, kernel_size**2, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        kernels = self.kernel_generator(x).view(
            B, 1, self.kernel_size, self.kernel_size
        )
        x_mean = x.mean(dim=1, keepdim=True)
        x_mean = x_mean.view(1, B, H, W)
        kernels = kernels.view(B, 1, self.kernel_size, self.kernel_size)
        att = F.conv2d(x_mean, weight=kernels, padding=self.kernel_size // 2, groups=B)
        att = att.view(B, 1, H, W)
        att = self.sigmoid(att)
        return x * att


class RCSAB(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        reduction,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super(RCSAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(ChannelAttention())
        modules_body.append(DynamicSpatialAttention())
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class DynamicResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(DynamicResidualGroup, self).__init__()
        modules_body = [
            RCSAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=True,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                res_scale=1,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )
