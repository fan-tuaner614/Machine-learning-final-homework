"""
步骤5: 模型评估模块
Model Evaluation and Analysis
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from typing import Dict, Tuple, Optional
import json


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 设备
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, X: np.ndarray, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测
        
        Args:
            X: 输入特征
            batch_size: 批次大小
            
        Returns:
            (预测标签, 预测概率)
        """
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)
    
    def evaluate(self,
                X_test: np.ndarray,
                y_test: np.ndarray,
                label_mapping: Dict,
                batch_size: int = 64,
                verbose: bool = True) -> Dict:
        """
        完整评估
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            label_mapping: 标签映射
            batch_size: 批次大小
            verbose: 是否打印详细信息
            
        Returns:
            评估结果字典
        """
        print("="*60)
        print("步骤5: 模型评估")
        print("="*60)
        
        # 预测
        y_pred, y_prob = self.predict(X_test, batch_size)
        
        # 计算各项指标
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 每个类别的指标
        idx_to_label = label_mapping['idx_to_label']
        class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
        
        # 分类报告
        report_dict = classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # 计算ROC-AUC（多分类）
        try:
            if len(np.unique(y_test)) > 2:
                # 多分类：使用one-vs-rest
                roc_auc = roc_auc_score(
                    y_test, y_prob,
                    multi_class='ovr',
                    average='weighted'
                )
            else:
                # 二分类
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        except:
            roc_auc = None
        
        # 打印结果
        if verbose:
            print(f"\n总体性能:")
            print(f"  准确率 (Accuracy):        {accuracy:.4f}")
            print(f"  精确率 (Precision):       {precision_weighted:.4f} (weighted), {precision_macro:.4f} (macro)")
            print(f"  召回率 (Recall):          {recall_weighted:.4f} (weighted), {recall_macro:.4f} (macro)")
            print(f"  F1分数 (F1-Score):        {f1_weighted:.4f} (weighted), {f1_macro:.4f} (macro)")
            if roc_auc is not None:
                print(f"  ROC-AUC:                  {roc_auc:.4f}")
            
            print(f"\n每个类别的性能:")
            print(f"{'类别':<15} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'样本数':>10}")
            print("-" * 60)
            for class_name in class_names:
                if class_name in report_dict:
                    metrics = report_dict[class_name]
                    print(f"{class_name:<15} "
                          f"{metrics['precision']:>10.4f} "
                          f"{metrics['recall']:>10.4f} "
                          f"{metrics['f1-score']:>10.4f} "
                          f"{metrics['support']:>10}")
            
            print(f"\n混淆矩阵:")
            print(f"{'':>15}", end='')
            for name in class_names:
                print(f"{name[:10]:>12}", end='')
            print()
            for i, name in enumerate(class_names):
                print(f"{name[:15]:>15}", end='')
                for j in range(len(class_names)):
                    print(f"{cm[i, j]:>12}", end='')
                print()
        
        # 构建结果字典
        results = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'precision_weighted': float(precision_weighted),
            'recall_macro': float(recall_macro),
            'recall_weighted': float(recall_weighted),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'confusion_matrix': cm.tolist(),
            'classification_report': report_dict,
            'predictions': y_pred.tolist(),
            'probabilities': y_prob.tolist(),
            'true_labels': y_test.tolist()
        }
        
        print(f"\n{'='*60}")
        print(f"评估完成！")
        print(f"{'='*60}\n")
        
        return results
    
    def analyze_errors(self,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      y_pred: np.ndarray,
                      label_mapping: Dict,
                      top_n: int = 10) -> Dict:
        """
        分析错误预测
        
        Args:
            X_test: 测试特征
            y_test: 真实标签
            y_pred: 预测标签
            label_mapping: 标签映射
            top_n: 显示前N个错误
            
        Returns:
            错误分析结果
        """
        idx_to_label = label_mapping['idx_to_label']
        
        # 找出错误的样本
        error_mask = y_test != y_pred
        error_indices = np.where(error_mask)[0]
        
        print(f"\n错误分析:")
        print(f"  总样本数: {len(y_test)}")
        print(f"  错误样本数: {len(error_indices)}")
        print(f"  错误率: {len(error_indices)/len(y_test)*100:.2f}%")
        
        # 统计每种错误类型
        error_types = {}
        for idx in error_indices:
            true_label = idx_to_label[y_test[idx]]
            pred_label = idx_to_label[y_pred[idx]]
            error_key = f"{true_label} → {pred_label}"
            error_types[error_key] = error_types.get(error_key, 0) + 1
        
        # 按错误数量排序
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n最常见的错误类型 (前{min(top_n, len(sorted_errors))}个):")
        for i, (error_type, count) in enumerate(sorted_errors[:top_n], 1):
            print(f"  {i}. {error_type}: {count} 次 ({count/len(error_indices)*100:.1f}%)")
        
        return {
            'n_errors': int(len(error_indices)),
            'error_rate': float(len(error_indices) / len(y_test)),
            'error_types': error_types,
            'error_indices': error_indices.tolist()
        }
    
    def compute_class_confidence(self,
                                 y_prob: np.ndarray,
                                 y_test: np.ndarray,
                                 label_mapping: Dict) -> Dict:
        """
        计算每个类别的预测置信度
        
        Args:
            y_prob: 预测概率
            y_test: 真实标签
            label_mapping: 标签映射
            
        Returns:
            置信度统计
        """
        idx_to_label = label_mapping['idx_to_label']
        n_classes = len(idx_to_label)
        
        confidence_stats = {}
        
        for class_idx in range(n_classes):
            class_name = idx_to_label[class_idx]
            
            # 这个类别的样本
            class_mask = y_test == class_idx
            if class_mask.sum() == 0:
                continue
            
            class_probs = y_prob[class_mask, class_idx]
            
            confidence_stats[class_name] = {
                'mean_confidence': float(np.mean(class_probs)),
                'std_confidence': float(np.std(class_probs)),
                'min_confidence': float(np.min(class_probs)),
                'max_confidence': float(np.max(class_probs)),
                'median_confidence': float(np.median(class_probs))
            }
        
        print(f"\n各类别预测置信度:")
        print(f"{'类别':<15} {'平均':>10} {'标准差':>10} {'最小':>10} {'最大':>10}")
        print("-" * 60)
        for class_name, stats in confidence_stats.items():
            print(f"{class_name:<15} "
                  f"{stats['mean_confidence']:>10.4f} "
                  f"{stats['std_confidence']:>10.4f} "
                  f"{stats['min_confidence']:>10.4f} "
                  f"{stats['max_confidence']:>10.4f}")
        
        return confidence_stats
    
    def save_results(self, results: Dict, save_path: str):
        """保存评估结果"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] 评估结果已保存到: {save_path}")
    
    def get_feature_importance(self,
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               feature_names: Optional[list] = None,
                               method: str = 'gradient') -> np.ndarray:
        """
        计算特征重要性（简化版）
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            feature_names: 特征名称列表
            method: 方法 ('gradient' 或 'permutation')
            
        Returns:
            特征重要性数组
        """
        if method == 'gradient':
            # 使用梯度作为重要性
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            X_tensor.requires_grad = True
            
            outputs = self.model(X_tensor)
            
            # 对预测类别的输出求梯度
            importance = []
            for i in range(outputs.shape[1]):
                self.model.zero_grad()
                outputs[:, i].sum().backward(retain_graph=True)
                importance.append(X_tensor.grad.abs().mean(0).cpu().detach().numpy())
            
            importance = np.mean(importance, axis=0)
            
        else:
            # 简化的排列重要性
            baseline_acc = accuracy_score(y_test, self.predict(X_test)[0])
            importance = np.zeros(X_test.shape[1])
            
            for i in range(X_test.shape[1]):
                X_permuted = X_test.copy()
                np.random.shuffle(X_permuted[:, i])
                permuted_acc = accuracy_score(y_test, self.predict(X_permuted)[0])
                importance[i] = baseline_acc - permuted_acc
        
        return importance


def main():
    """测试评估模块"""
    import sys
    sys.path.append('.')
    from models.neural_network import CellTypeClassifier
    
    # 创建模拟数据
    n_samples = 200
    input_dim = 50
    n_classes = 4
    
    X_test = np.random.randn(n_samples, input_dim).astype(np.float32)
    y_test = np.random.randint(0, n_classes, n_samples)
    
    # 创建模拟标签映射
    label_mapping = {
        'idx_to_label': {0: 'ER+', 1: 'ER+HER2+', 2: 'HER2+', 3: 'TN'},
        'label_to_idx': {'ER+': 0, 'ER+HER2+': 1, 'HER2+': 2, 'TN': 3},
        'n_classes': 4
    }
    
    # 创建模型
    model = CellTypeClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_dims=[128, 64],
        dropout_rate=0.3
    )
    
    # 创建评估器
    evaluator = ModelEvaluator(model=model, device='cuda')
    
    # 评估
    results = evaluator.evaluate(
        X_test, y_test,
        label_mapping=label_mapping,
        batch_size=64
    )
    
    # 预测
    y_pred, y_prob = evaluator.predict(X_test)
    
    # 错误分析
    error_analysis = evaluator.analyze_errors(
        X_test, y_test, y_pred,
        label_mapping=label_mapping
    )
    
    # 置信度分析
    confidence_stats = evaluator.compute_class_confidence(
        y_prob, y_test,
        label_mapping=label_mapping
    )


if __name__ == '__main__':
    main()
