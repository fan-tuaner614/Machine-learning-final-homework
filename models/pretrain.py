"""
自监督预训练模块
实现自编码器(AutoEncoder)和SimCLR对比学习模型
用于在SC-2全量数据上进行无监督预训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AutoEncoder(nn.Module):
    """
    自编码器模型
    用于学习单细胞数据的低维表示
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], latent_dim=64):
        super(AutoEncoder, self).__init__()
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # 潜在层
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """编码：提取特征"""
        return self.encoder(x)
    
    def decode(self, z):
        """解码：重构输入"""
        return self.decoder(z)
    
    def forward(self, x):
        """前向传播"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class ProjectionHead(nn.Module):
    """
    SimCLR的投影头
    将编码特征映射到对比学习空间
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """
    SimCLR对比学习模型
    通过对比损失学习有区分性的特征表示
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], latent_dim=64, projection_dim=64):
        super(SimCLR, self).__init__()
        
        # 编码器（与AutoEncoder相同结构）
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 投影头（仅用于对比学习训练）
        self.projection_head = ProjectionHead(latent_dim, hidden_dim=128, output_dim=projection_dim)
        
    def encode(self, x):
        """编码：提取特征（不包括投影头）"""
        return self.encoder(x)
    
    def forward(self, x):
        """前向传播：返回编码特征和投影特征"""
        h = self.encode(x)
        z = self.projection_head(h)
        return h, z


class ContrastiveLoss(nn.Module):
    """
    NT-Xent对比损失函数
    用于SimCLR训练
    """
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        """
        计算对比损失
        z_i, z_j: 同一batch的两个增强视图的特征，shape: (batch_size, projection_dim)
        """
        batch_size = z_i.shape[0]
        
        # 归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 拼接
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, projection_dim)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(z, z.T) / self.temperature  # (2*batch_size, 2*batch_size)
        
        # 创建正样本mask
        positive_mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=z.device)
        for i in range(batch_size):
            positive_mask[i, batch_size + i] = True
            positive_mask[batch_size + i, i] = True
        
        # 创建负样本mask（排除自己）
        negative_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        
        # 计算损失
        exp_sim = torch.exp(sim_matrix)
        
        # 对每个样本，正样本相似度 / (所有负样本相似度之和)
        loss = 0
        for i in range(2 * batch_size):
            positive_sim = exp_sim[i][positive_mask[i]].sum()
            negative_sim = exp_sim[i][negative_mask[i]].sum()
            loss += -torch.log(positive_sim / negative_sim)
        
        loss = loss / (2 * batch_size)
        return loss


class SingleCellDataset(Dataset):
    """
    单细胞数据集
    支持自监督学习的数据增强
    """
    def __init__(self, data, augment=False):
        """
        data: numpy array, shape (n_cells, n_features)
        augment: 是否进行数据增强（用于SimCLR）
        """
        self.data = torch.FloatTensor(data)
        self.augment = augment
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        
        if self.augment:
            # 数据增强：添加高斯噪声
            x1 = x + torch.randn_like(x) * 0.1
            x2 = x + torch.randn_like(x) * 0.1
            
            # 随机Dropout
            mask1 = torch.rand_like(x) > 0.1
            mask2 = torch.rand_like(x) > 0.1
            x1 = x1 * mask1
            x2 = x2 * mask2
            
            return x1, x2
        else:
            return x
    

def train_autoencoder(model, dataloader, optimizer, device, epochs=100):
    """
    训练自编码器
    """
    model.train()
    criterion = nn.MSELoss()
    
    history = {'loss': []}
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            
            # 前向传播
            x_recon, z = model(batch_data)
            loss = criterion(x_recon, batch_data)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return history


def train_simclr(model, dataloader, optimizer, device, epochs=100, temperature=0.5):
    """
    训练SimCLR模型
    """
    model.train()
    criterion = ContrastiveLoss(temperature=temperature)
    
    history = {'loss': []}
    
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2 in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            
            # 前向传播
            _, z1 = model(x1)
            _, z2 = model(x2)
            
            loss = criterion(z1, z2)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Contrastive Loss: {avg_loss:.4f}")
    
    return history


def extract_features(model, dataloader, device):
    """
    使用训练好的模型提取特征
    """
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                batch_data = batch_data[0]  # SimCLR返回两个视图，只取第一个
            
            batch_data = batch_data.to(device)
            
            if isinstance(model, AutoEncoder):
                z = model.encode(batch_data)
            elif isinstance(model, SimCLR):
                z = model.encode(batch_data)
            
            features.append(z.cpu().numpy())
    
    return np.concatenate(features, axis=0)
