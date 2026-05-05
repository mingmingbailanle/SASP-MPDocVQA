#SASP（Self-Attention Scoring Page Selection）
import glob
import cv2
from PIL import Image
import os
import numpy as np
import time
import random
from transformers import DistilBertModel
import torch

# 控制是否使用简化模式（True时使用简单的池化，False时使用Transformer编码器）
FACIL = False

# 自动检测并使用可用的计算设备（优先使用GPU，否则使用CPU）
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ProbModule(torch.nn.Module):
    #概率预测模块：用于处理序列特征并输出概率分数的神经网络
    def __init__(self):
        #初始化
        super(ProbModule, self).__init__()

        # 自适应平均池化层：将任意长度序列压缩为固定长度1
        # 用于FACIL=True时的简单特征聚合
        self.avg = torch.nn.AdaptiveAvgPool1d(1)

        # ========== 优化：全连接层加宽加深（两个分支共用）==========
        # 768→512→256→128→64→1（5层）
        self.fc0 = torch.nn.Linear(768, 512)   # 256→512，加宽
        self.fc1 = torch.nn.Linear(512, 256)   # 新增中间层
        self.fc2 = torch.nn.Linear(256, 128)   # 64→128，扩展
        self.fc3 = torch.nn.Linear(128, 64)    # 新增中间层
        self.fc4 = torch.nn.Linear(64, 1)      # 输出层

        # Dropout正则化：随机丢弃30%的神经元
        self.drop = torch.nn.Dropout(p=0.3)

        # 激活函数升级
        self.gelu = torch.nn.GELU()            # 新增GELU（深层网络标配）
        self.relu = torch.nn.LeakyReLU(0.1)    # 负斜率0.01→0.1，增强梯度

        # Transformer编码器层配置（仅FACIL=False时使用）
        # 1层Transformer，FFN维度1536，Dropout0.3，GELU激活
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=768,
            nhead=16,
            dim_feedforward=1536,      # 768→1536（2倍扩展）
            dropout=0.3,               # 0.3
            batch_first=True,
            activation='gelu'          # ReLU→GELU
        )

        # 构建双层Transformer编码器（原1层→2层）
        self.enc = torch.nn.TransformerEncoder(enc_layer, num_layers=1).to(DEVICE)

    def forward(self, feat, mask):
        """
        参数:
            feat: 输入特征张量，形状为 (batch_size, channel, length)
                  例如: (batch, 768, 2048) 表示batch个样本，每个样本2048个时间步，每步768维特征
            mask: 掩码张量，形状为 (batch_size, length)
                  用于标记哪些位置是有效特征(1.0)，哪些是填充PAD(0.0)
                  用于标记哪些位置是有效特征(1.0)，哪些是填充PAD(0.0)
        返回:
            res: 概率分数，形状为 (batch_size,)
        """

        # ==================== 分支1: 简化模式（FACIL=True）====================
        if FACIL:
            # 使用简单的特征处理流程
            out = self.relu(feat)       # LeakyReLU激活（负斜率0.1）
            out = self.drop(out)        # Dropout 0.25
            out = self.avg(out)         # 自适应平均池化
            oout = out.squeeze(-1)      # (batch, 768)

        # ==================== 分支2: Transformer模式（FACIL=False，默认）====================
        else:
            # 将mask转换为布尔类型的关键填充掩码
            bin_mask = (mask < 0.5)

            # 调整特征维度顺序以适应Transformer
            permuted_feat = torch.permute(feat, (0, 2, 1))

            # 单层Transformer编码器处理
            out = self.enc(permuted_feat, src_key_padding_mask=bin_mask)

            # 提取CLS位置特征
            oout = out[:, 0, :]         # (batch, 768)

        # ==================== 后续全连接层（两个分支共用）====================

        # 加深的5层全连接网络
        res = self.gelu(oout)       # 第一层用GELU（平滑梯度）
        res = self.drop(res)
        res = self.fc0(res)         # 768 → 512

        res = self.gelu(res)
        res = self.drop(res)
        res = self.fc1(res)         # 512 → 256

        res = self.gelu(res)
        res = self.drop(res)
        res = self.fc2(res)         # 256 → 128

        res = self.relu(res)        # 混合激活：LeakyReLU
        res = self.drop(res)
        res = self.fc3(res)         # 128 → 64

        res = self.fc4(res)         # 64 → 1

        # 移除最后一个维度，返回概率分数
        return torch.sigmoid(res).squeeze(-1)
