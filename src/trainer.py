"""
步骤4: 模型训练模块
Training Module with GPU Support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import time
from tqdm import tqdm


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0001, mode: str = 'max'):
        """
        初始化早停
        
        Args:
            patience: 容忍轮数
            min_delta: 最小改进量
            mode: 'max'表示越大越好，'min'表示越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 optimizer_type: str = 'adamw',
                 scheduler_type: str = 'reduce_lr',
                 loss_type: str = 'ce',
                 class_weights: Optional[torch.Tensor] = None):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            device: 设备 ('cuda' 或 'cpu')
            learning_rate: 学习率
            weight_decay: 权重衰减
            optimizer_type: 优化器类型
            scheduler_type: 学习率调度器类型
            loss_type: 损失函数类型
            class_weights: 类别权重
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        
        # 显示设备信息
        if self.device.type == 'cuda':
            print(f"\n{'='*60}")
            print(f"使用GPU训练")
            print(f"  设备: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"{'='*60}\n")
        else:
            print(f"\n⚠ 警告: CUDA不可用，使用CPU训练\n")
        
        # 优化器
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"未知的优化器类型: {optimizer_type}")
        
        # 损失函数
        if loss_type == 'ce':
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == 'focal':
            from models.neural_network import FocalLoss
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
            self.criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        elif loss_type == 'label_smoothing':
            from models.neural_network import LabelSmoothingLoss
            n_classes = model.n_classes
            self.criterion = LabelSmoothingLoss(n_classes, smoothing=0.1)
        else:
            raise ValueError(f"未知的损失函数类型: {loss_type}")
        
        # 学习率调度器
        self.scheduler_type = scheduler_type
        self.scheduler = None
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def create_data_loaders(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """
        创建数据加载器
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            batch_size: 批次大小
            
        Returns:
            (训练加载器, 验证加载器)
        """
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Windows下设置为0
            pin_memory=(self.device.type == 'cuda')
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == 'cuda')
        )
        
        return train_loader, val_loader
    
    def _setup_scheduler(self, train_loader: DataLoader, n_epochs: int):
        """设置学习率调度器"""
        if self.scheduler_type == 'reduce_lr':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True,
                min_lr=1e-7
            )
        elif self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=n_epochs,
                eta_min=1e-7
            )
        elif self.scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate * 10,
                steps_per_epoch=len(train_loader),
                epochs=n_epochs
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # OneCycle调度器在每个batch后更新
            if self.scheduler_type == 'onecycle' and self.scheduler is not None:
                self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             n_epochs: int = 100,
             batch_size: int = 64,
             early_stopping_patience: int = 15,
             verbose: bool = True) -> Dict:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            n_epochs: 训练轮数
            batch_size: 批次大小
            early_stopping_patience: 早停容忍轮数
            verbose: 是否显示详细信息
            
        Returns:
            训练历史字典
        """
        print("="*60)
        print("步骤4: 模型训练")
        print("="*60)
        
        print(f"\n训练配置:")
        print(f"  训练样本: {len(X_train)}")
        print(f"  验证样本: {len(X_val)}")
        print(f"  批次大小: {batch_size}")
        print(f"  训练轮数: {n_epochs}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  优化器: {self.optimizer.__class__.__name__}")
        print(f"  损失函数: {self.criterion.__class__.__name__}")
        print(f"  调度器: {self.scheduler_type}")
        print(f"  早停容忍: {early_stopping_patience} 轮")
        
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # 设置学习率调度器
        self._setup_scheduler(train_loader, n_epochs)
        
        # 早停
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.0001,
            mode='max'
        )
        
        print(f"\n{'='*60}")
        print("开始训练...")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # 训练循环
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # 更新学习率（非OneCycle）
            if self.scheduler_type == 'reduce_lr' and self.scheduler is not None:
                self.scheduler.step(val_acc)
            elif self.scheduler_type == 'cosine' and self.scheduler is not None:
                self.scheduler.step()
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                is_best = True
            else:
                is_best = False
            
            epoch_time = time.time() - epoch_start
            
            # 打印进度
            if verbose and ((epoch + 1) % 5 == 0 or is_best or epoch == 0):
                print(f"Epoch [{epoch+1:3d}/{n_epochs}] "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                      f"Time: {epoch_time:.1f}s"
                      f"{' ★ 最佳' if is_best else ''}")
            
            # 早停检查
            if early_stopping(val_acc):
                print(f"\n早停触发！在第 {epoch+1} 轮停止训练")
                break
        
        total_time = time.time() - start_time
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\n{'='*60}")
        print(f"训练完成！")
        print(f"  最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"  总训练时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        print(f"  平均每轮时间: {total_time/(epoch+1):.1f}秒")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_model(self, path: str):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }, path)
        print(f"[OK] 模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint['best_val_acc']
        print(f"[OK] 模型已从 {path} 加载")


def main():
    """测试训练模块"""
    import sys
    sys.path.append('.')
    from models.neural_network import CellTypeClassifier
    
    # 创建模拟数据
    n_samples = 1000
    input_dim = 50
    n_classes = 4
    
    X_train = np.random.randn(n_samples, input_dim).astype(np.float32)
    y_train = np.random.randint(0, n_classes, n_samples)
    X_val = np.random.randn(200, input_dim).astype(np.float32)
    y_val = np.random.randint(0, n_classes, 200)
    
    # 创建模型
    model = CellTypeClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_dims=[128, 64],
        dropout_rate=0.3
    )
    
    # 创建训练器
    trainer = ModelTrainer(
        model=model,
        device='cuda',
        learning_rate=0.001,
        optimizer_type='adamw',
        scheduler_type='reduce_lr',
        loss_type='ce'
    )
    
    # 训练
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        n_epochs=20,
        batch_size=64,
        early_stopping_patience=5
    )
    
    print("\n训练历史:")
    print(f"  最终训练准确率: {history['train_acc'][-1]:.4f}")
    print(f"  最终验证准确率: {history['val_acc'][-1]:.4f}")
    print(f"  最佳验证准确率: {max(history['val_acc']):.4f}")


if __name__ == '__main__':
    main()
