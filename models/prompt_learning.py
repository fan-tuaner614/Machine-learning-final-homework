"""
提示学习框架模块
实现冻结预训练编码器 + 可学习提示向量 + 轻量级分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PromptLearningModel(nn.Module):
    """
    提示学习模型
    冻结预训练编码器，只训练提示向量和分类头
    """
    def __init__(self, pretrained_encoder, latent_dim, num_classes, 
                 prompt_length=10, prompt_dim=None, dropout=0.1):
        """
        Args:
            pretrained_encoder: 预训练的编码器模型
            latent_dim: 编码器输出的特征维度
            num_classes: 分类类别数
            prompt_length: 提示向量的数量
            prompt_dim: 提示向量的维度（如果为None，则与latent_dim相同）
            dropout: Dropout率
        """
        super(PromptLearningModel, self).__init__()
        
        # 冻结预训练编码器
        self.encoder = pretrained_encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 可学习的提示向量
        if prompt_dim is None:
            prompt_dim = latent_dim
        
        self.prompt_length = prompt_length
        self.prompt_dim = prompt_dim
        
        # 初始化提示向量（可学习参数）
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, prompt_dim) * 0.01
        )
        
        # 如果prompt_dim != latent_dim，需要投影
        self.need_projection = (prompt_dim != latent_dim)
        if self.need_projection:
            self.prompt_projection = nn.Linear(prompt_dim, latent_dim)
        
        # 轻量级分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim + latent_dim * prompt_length),  # 拼接后的维度
            nn.Dropout(dropout),
            nn.Linear(latent_dim + latent_dim * prompt_length, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        前向传播
        x: 输入数据 (batch_size, input_dim)
        """
        batch_size = x.shape[0]
        
        # 1. 通过冻结的编码器提取特征
        with torch.no_grad():
            encoded_features = self.encoder(x)  # (batch_size, latent_dim)
        
        # 2. 获取提示向量并投影（如果需要）
        if self.need_projection:
            prompt_features = self.prompt_projection(self.prompt_embeddings)  # (prompt_length, latent_dim)
        else:
            prompt_features = self.prompt_embeddings  # (prompt_length, latent_dim)
        
        # 3. 扩展提示向量到batch
        prompt_features = prompt_features.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, prompt_length, latent_dim)
        
        # 4. 拼接编码特征和提示特征
        # encoded_features: (batch_size, latent_dim)
        # prompt_features: (batch_size, prompt_length, latent_dim)
        encoded_features = encoded_features.unsqueeze(1)  # (batch_size, 1, latent_dim)
        
        # 拼接方式：将prompt_features展平后与encoded_features拼接
        prompt_features_flat = prompt_features.reshape(batch_size, -1)  # (batch_size, prompt_length * latent_dim)
        encoded_features_flat = encoded_features.squeeze(1)  # (batch_size, latent_dim)
        
        combined_features = torch.cat([encoded_features_flat, prompt_features_flat], dim=1)  # (batch_size, latent_dim + prompt_length * latent_dim)
        
        # 5. 通过分类头
        logits = self.classifier(combined_features)
        
        return logits
    
    def get_trainable_parameters(self):
        """返回可训练的参数"""
        trainable_params = []
        trainable_params.append({'params': self.prompt_embeddings})
        
        if self.need_projection:
            trainable_params.append({'params': self.prompt_projection.parameters()})
        
        trainable_params.append({'params': self.classifier.parameters()})
        
        return trainable_params
    
    def count_parameters(self):
        """统计参数数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'trainable_ratio': trainable / total if total > 0 else 0
        }


class LabeledSubsetDataset(Dataset):
    """
    有标签子集数据集
    根据索引从完整数据中提取有标签样本
    """
    def __init__(self, data, labels, indices=None):
        """
        Args:
            data: numpy array, shape (n_cells, n_features)
            labels: numpy array, shape (n_cells,)
            indices: numpy array, 要使用的样本索引
        """
        if indices is not None:
            self.data = torch.FloatTensor(data[indices])
            self.labels = torch.LongTensor(labels[indices])
        else:
            self.data = torch.FloatTensor(data)
            self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_prompt_model(model, train_loader, val_loader, optimizer, device, 
                       epochs=50, patience=10, verbose=True):
    """
    训练提示学习模型
    
    Args:
        model: PromptLearningModel实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 设备
        epochs: 训练轮数
        patience: 早停耐心值
        verbose: 是否打印详细信息
    
    Returns:
        history: 训练历史
        best_model_state: 最佳模型状态
    """
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # 前向传播
            logits = model(batch_data)
            loss = criterion(logits, batch_labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * batch_data.size(0)
            _, predicted = torch.max(logits, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                logits = model(batch_data)
                loss = criterion(logits, batch_labels)
                
                val_loss += loss.item() * batch_data.size(0)
                _, predicted = torch.max(logits, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印信息
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    return history, best_model_state


def evaluate_prompt_model(model, test_loader, device):
    """
    评估提示学习模型
    
    Returns:
        accuracy, predictions, true_labels
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(batch_data)
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = (all_predictions == all_labels).mean()
    
    return accuracy, all_predictions, all_labels


def extract_prompt_features(model, dataloader, device):
    """
    使用提示学习模型提取特征（编码器输出+提示向量）
    
    Returns:
        features: numpy array, shape (n_samples, feature_dim)
    """
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)
            batch_size = batch_data.shape[0]
            
            # 提取编码特征
            encoded_features = model.encoder(batch_data)  # (batch_size, latent_dim)
            
            # 获取提示特征
            if model.need_projection:
                prompt_features = model.prompt_projection(model.prompt_embeddings)
            else:
                prompt_features = model.prompt_embeddings
            
            # 扩展并拼接
            prompt_features = prompt_features.unsqueeze(0).expand(batch_size, -1, -1)
            prompt_features_flat = prompt_features.reshape(batch_size, -1)
            
            combined_features = torch.cat([encoded_features, prompt_features_flat], dim=1)
            
            all_features.append(combined_features.cpu().numpy())
    
    return np.concatenate(all_features, axis=0)
