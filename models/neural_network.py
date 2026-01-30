"""
步骤3: 深度学习模型定义
Neural Network Models for Single-Cell Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class AttentionLayer(nn.Module):
    """注意力层"""
    
    def __init__(self, input_dim: int, attention_dim: int = 128):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
    
    def forward(self, x):
        # 计算注意力权重
        attention_weights = torch.softmax(self.attention(x), dim=1)
        # 应用注意力
        attended = x * attention_weights
        return attended, attention_weights


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, dim: int, dropout_rate: float = 0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class CellTypeClassifier(nn.Module):
    """细胞类型分类器 - 基础版"""
    
    def __init__(self,
                 input_dim: int,
                 n_classes: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True):
        """
        初始化分类器
        
        Args:
            input_dim: 输入特征维度
            n_classes: 类别数量
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout比率
            use_batch_norm: 是否使用BatchNorm
        """
        super(CellTypeClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 分类层
        self.classifier = nn.Linear(prev_dim, n_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """获取特征表示"""
        return self.feature_extractor(x)


class AttentionCellClassifier(nn.Module):
    """带注意力机制的细胞分类器"""
    
    def __init__(self,
                 input_dim: int,
                 n_classes: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 attention_dim: int = 128):
        """
        初始化带注意力的分类器
        
        Args:
            input_dim: 输入特征维度
            n_classes: 类别数量
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout比率
            attention_dim: 注意力层维度
        """
        super(AttentionCellClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # 特征提取器
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 注意力层
        self.attention = AttentionLayer(prev_dim, attention_dim)
        
        # 分类器
        self.classifier = nn.Linear(prev_dim, n_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        attended_features, attention_weights = self.attention(features)
        logits = self.classifier(attended_features)
        return logits
    
    def forward_with_attention(self, x):
        """返回预测结果和注意力权重"""
        features = self.feature_extractor(x)
        attended_features, attention_weights = self.attention(features)
        logits = self.classifier(attended_features)
        return logits, attention_weights


class ResidualCellClassifier(nn.Module):
    """带残差连接的细胞分类器"""
    
    def __init__(self,
                 input_dim: int,
                 n_classes: int,
                 hidden_dim: int = 256,
                 n_residual_blocks: int = 3,
                 dropout_rate: float = 0.3):
        """
        初始化残差分类器
        
        Args:
            input_dim: 输入特征维度
            n_classes: 类别数量
            hidden_dim: 隐藏层维度
            n_residual_blocks: 残差块数量
            dropout_rate: Dropout比率
        """
        super(ResidualCellClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate)
            for _ in range(n_residual_blocks)
        ])
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, n_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        logits = self.classifier(x)
        return logits


class FocalLoss(nn.Module):
    """Focal Loss - 用于处理类别不平衡"""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha: 类别权重
            gamma: 聚焦参数
            reduction: 损失聚合方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    
    def __init__(self, n_classes: int, smoothing: float = 0.1):
        """
        初始化标签平滑损失
        
        Args:
            n_classes: 类别数量
            smoothing: 平滑系数
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def create_model(model_type: str,
                 input_dim: int,
                 n_classes: int,
                 **kwargs) -> nn.Module:
    """
    创建模型的工厂函数
    
    Args:
        model_type: 模型类型 ('basic', 'attention', 'residual')
        input_dim: 输入维度
        n_classes: 类别数量
        **kwargs: 其他模型参数
        
    Returns:
        模型实例
    """
    if model_type == 'basic':
        return CellTypeClassifier(input_dim, n_classes, **kwargs)
    elif model_type == 'attention':
        return AttentionCellClassifier(input_dim, n_classes, **kwargs)
    elif model_type == 'residual':
        return ResidualCellClassifier(input_dim, n_classes, **kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")


def main():
    """测试模型定义"""
    # 测试参数
    input_dim = 50
    n_classes = 4
    batch_size = 32
    
    print("="*60)
    print("测试模型定义")
    print("="*60)
    
    # 创建测试数据
    x = torch.randn(batch_size, input_dim)
    
    # 测试基础模型
    print("\n1. 基础分类器")
    model_basic = CellTypeClassifier(input_dim, n_classes)
    out = model_basic(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out.shape}")
    print(f"   参数量: {sum(p.numel() for p in model_basic.parameters()):,}")
    
    # 测试注意力模型
    print("\n2. 注意力分类器")
    model_attention = AttentionCellClassifier(input_dim, n_classes)
    out = model_attention(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out.shape}")
    print(f"   参数量: {sum(p.numel() for p in model_attention.parameters()):,}")
    
    # 测试残差模型
    print("\n3. 残差分类器")
    model_residual = ResidualCellClassifier(input_dim, n_classes)
    out = model_residual(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out.shape}")
    print(f"   参数量: {sum(p.numel() for p in model_residual.parameters()):,}")
    
    # 测试损失函数
    print("\n4. 测试损失函数")
    targets = torch.randint(0, n_classes, (batch_size,))
    
    # Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(out, targets)
    print(f"   Focal Loss: {loss.item():.4f}")
    
    # Label Smoothing
    ls_loss = LabelSmoothingLoss(n_classes, smoothing=0.1)
    loss = ls_loss(out, targets)
    print(f"   Label Smoothing Loss: {loss.item():.4f}")
    
    print("\n" + "="*60)
    print("模型测试完成！")
    print("="*60)


if __name__ == '__main__':
    main()
