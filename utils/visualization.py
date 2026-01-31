"""可视化工具模块
Visualization Utilities
"""

import matplotlib

# 使用非交互式后端（必须在导入 pyplot 之前设置）
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
import os
warnings.filterwarnings('ignore')

# 导入中文字体配置
from utils.chinese_font import setup_chinese_font, get_chinese_font
setup_chinese_font()

sns.set_style('whitegrid')


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, output_dir: str = 'outputs'):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font_prop = get_chinese_font()
    
    def plot_training_history(self,
                              history: Dict,
                              dataset_name: str,
                              save: bool = True):
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典
            dataset_name: 数据集名称
            save: 是否保存
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        font_prop = self.font_prop
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12, fontproperties=font_prop)
        axes[0, 0].set_ylabel('Loss', fontsize=12, fontproperties=font_prop)
        axes[0, 0].set_title('损失曲线', fontsize=14, fontweight='bold', fontproperties=font_prop)
        axes[0, 0].legend(fontsize=10, prop=font_prop)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12, fontproperties=font_prop)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12, fontproperties=font_prop)
        axes[0, 1].set_title('准确率曲线', fontsize=14, fontweight='bold', fontproperties=font_prop)
        axes[0, 1].legend(fontsize=10, prop=font_prop)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率曲线（如果有）
        if 'lr' in history and len(history['lr']) > 0:
            axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=12, fontproperties=font_prop)
            axes[1, 0].set_ylabel('Learning Rate', fontsize=12, fontproperties=font_prop)
            axes[1, 0].set_title('学习率变化', fontsize=14, fontweight='bold', fontproperties=font_prop)
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # 如果没有学习率数据，显示损失差距
            loss_gap = [train - val for train, val in zip(history['train_loss'], history['val_loss'])]
            axes[1, 0].plot(epochs, loss_gap, 'orange', linewidth=2)
            axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1, 0].set_xlabel('Epoch', fontsize=12, fontproperties=font_prop)
            axes[1, 0].set_ylabel('Train Loss - Val Loss', fontsize=12, fontproperties=font_prop)
            axes[1, 0].set_title('损失差距', fontsize=14, fontweight='bold', fontproperties=font_prop)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 训练验证差距
        acc_gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
        axes[1, 1].plot(epochs, acc_gap, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch', fontsize=12, fontproperties=font_prop)
        axes[1, 1].set_ylabel('Train Acc - Val Acc', fontsize=12, fontproperties=font_prop)
        axes[1, 1].set_title('过拟合程度 (越接近0越好)', fontsize=14, fontweight='bold', fontproperties=font_prop)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{dataset_name} - 训练历史', fontsize=16, fontweight='bold', y=0.995, fontproperties=font_prop)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'{dataset_name}_training_history.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
            print(f"[OK] 保存训练历史图: {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self,
                             cm: np.ndarray,
                             class_names: List[str],
                             dataset_name: str,
                             normalize: bool = True,
                             save: bool = True):
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            class_names: 类别名称
            dataset_name: 数据集名称
            normalize: 是否归一化
            save: 是否保存
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = '归一化混淆矩阵'
        else:
            fmt = 'd'
            title = '混淆矩阵'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': '比例' if normalize else '数量'},
            vmin=0,
            vmax=1 if normalize else None
        )
        
        plt.title(f'{dataset_name} - {title}', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'{dataset_name}_confusion_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
            print(f"[OK] 保存混淆矩阵: {save_path}")
        
        plt.close()
    
    def plot_class_performance(self,
                               report_dict: Dict,
                               class_names: List[str],
                               dataset_name: str,
                               save: bool = True):
        """
        绘制各类别性能对比
        
        Args:
            report_dict: 分类报告字典
            class_names: 类别名称
            dataset_name: 数据集名称
            save: 是否保存
        """
        metrics = ['precision', 'recall', 'f1-score']
        data = {metric: [] for metric in metrics}
        
        for class_name in class_names:
            if class_name in report_dict:
                for metric in metrics:
                    data[metric].append(report_dict[class_name][metric])
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            offset = width * (i - 1)
            ax.bar(x + offset, data[metric], width, label=metric.capitalize())
        
        ax.set_xlabel('类别', fontsize=12)
        ax.set_ylabel('分数', fontsize=12)
        ax.set_title(f'{dataset_name} - 各类别性能对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'{dataset_name}_class_performance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
            print(f"[OK] 保存类别性能图: {save_path}")
        
        plt.close()
    
    def plot_probability_distribution(self,
                                      y_prob: np.ndarray,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      class_names: List[str],
                                      dataset_name: str,
                                      save: bool = True):
        """
        绘制预测概率分布
        
        Args:
            y_prob: 预测概率
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            dataset_name: 数据集名称
            save: 是否保存
        """
        n_classes = len(class_names)
        fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()
        
        for i, class_name in enumerate(class_names):
            ax = axes[i]
            
            # 正确预测的概率
            correct_mask = (y_true == i) & (y_pred == i)
            correct_probs = y_prob[correct_mask, i] if correct_mask.any() else []
            
            # 错误预测的概率
            incorrect_mask = (y_true == i) & (y_pred != i)
            incorrect_probs = y_prob[incorrect_mask, i] if incorrect_mask.any() else []
            
            if len(correct_probs) > 0:
                ax.hist(correct_probs, bins=20, alpha=0.6, label='正确', color='green', edgecolor='black')
            if len(incorrect_probs) > 0:
                ax.hist(incorrect_probs, bins=20, alpha=0.6, label='错误', color='red', edgecolor='black')
            
            ax.set_xlabel('预测概率', fontsize=10)
            ax.set_ylabel('样本数', fontsize=10)
            ax.set_title(f'{class_name}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{dataset_name} - 预测概率分布', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'{dataset_name}_probability_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
            print(f"[OK] 保存概率分布图: {save_path}")
        
        plt.close()
    
    def plot_model_comparison(self,
                             results_dict: Dict[str, Dict],
                             metric: str = 'accuracy',
                             save: bool = True):
        """
        绘制多个模型的对比
        
        Args:
            results_dict: 多个模型的结果字典
            metric: 对比的指标
            save: 是否保存
        """
        model_names = list(results_dict.keys())
        values = [results_dict[name][metric] for name in model_names]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(model_names)), values, color='steelblue', edgecolor='black')
        
        # 在柱子上标注数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('模型', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'模型性能对比 - {metric.capitalize()}', fontsize=14, fontweight='bold')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylim([0, max(values) * 1.2])
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'model_comparison_{metric}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
            print(f"[OK] 保存模型对比图: {save_path}")
        
        plt.close()


def main():
    """测试可视化模块"""
    visualizer = Visualizer(output_dir='outputs')
    
    # 测试训练历史图
    history = {
        'train_loss': [1.2, 1.0, 0.8, 0.6, 0.5],
        'val_loss': [1.3, 1.1, 0.9, 0.8, 0.7],
        'train_acc': [0.5, 0.6, 0.7, 0.8, 0.85],
        'val_acc': [0.48, 0.58, 0.68, 0.75, 0.78],
        'lr': [0.001, 0.001, 0.0005, 0.0005, 0.00025]
    }
    visualizer.plot_training_history(history, 'test_dataset')
    
    # 测试混淆矩阵
    cm = np.array([[50, 5, 3, 2], [8, 45, 4, 3], [6, 7, 40, 7], [4, 3, 8, 45]])
    class_names = ['ER+', 'ER+HER2+', 'HER2+', 'TN']
    visualizer.plot_confusion_matrix(cm, class_names, 'test_dataset')
    
    print("\n可视化测试完成！")


if __name__ == '__main__':
    main()
