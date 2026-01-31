"""
增强可视化模块 - 为实验报告生成美观的专业级图表
Enhanced Visualization Module - Generate Professional Publication-Quality Figures
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入中文字体配置
from utils.chinese_font import setup_chinese_font, get_chinese_font
setup_chinese_font()

# 设置全局样式
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


class EnhancedVisualizer:
    """增强可视化类 - 生成高质量专业图表"""
    
    # 专业配色方案
    COLORS = {
        'primary': '#2E86AB',      # 深蓝
        'secondary': '#A23B72',    # 玫红
        'accent': '#F18F01',       # 橙色
        'success': '#C73E1D',      # 红褐
        'info': '#3B1F2B',         # 深紫
        'encoder': '#3498db',      # 蓝色
        'decoder': '#2ecc71',      # 绿色
        'latent': '#e74c3c',       # 红色
        'frozen': '#95a5a6',       # 灰色
        'learnable': '#1abc9c',    # 青色
        'gradient': ['#667eea', '#764ba2'],  # 渐变
    }
    
    # 细胞亚型配色
    CELL_COLORS = {
        'ER+': '#E74C3C',
        'HER2+': '#3498DB', 
        'TN': '#2ECC71',
        'ER+HER2+': '#9B59B6',
        'Unknown': '#95A5A6',
    }
    
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font_prop = get_chinese_font()
    
    # ==================== 初级实验可视化 ====================
    
    def plot_autoencoder_training_curve_enhanced(
        self, 
        history: List[float],
        save_path: Optional[str] = None
    ):
        """
        绘制美观的自编码器训练曲线
        
        Args:
            history: 损失历史
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        epochs = range(1, len(history) + 1)
        
        # 主曲线 - 带渐变填充
        ax.fill_between(epochs, history, alpha=0.3, color=self.COLORS['primary'])
        ax.plot(epochs, history, linewidth=2.5, color=self.COLORS['primary'], 
                label='重构损失 (MSE)', marker='o', markersize=4, markevery=10)
        
        # 标注最低点
        min_loss = min(history)
        min_epoch = history.index(min_loss) + 1
        ax.scatter([min_epoch], [min_loss], s=150, c=self.COLORS['accent'], 
                   zorder=5, edgecolors='white', linewidths=2)
        ax.annotate(f'最低损失: {min_loss:.4f}\n(Epoch {min_epoch})',
                   xy=(min_epoch, min_loss), xytext=(min_epoch+10, min_loss+0.05),
                   fontsize=11, fontproperties=self.font_prop,
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['accent'], lw=1.5),
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor=self.COLORS['accent'], alpha=0.9))
        
        # 添加收敛区域标注
        converge_epoch = len(history) // 5  # 假设前20%收敛
        ax.axvspan(1, converge_epoch, alpha=0.1, color=self.COLORS['secondary'], 
                   label='快速下降阶段')
        ax.axvspan(converge_epoch, len(history), alpha=0.05, color=self.COLORS['success'],
                   label='稳定收敛阶段')
        
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=14, fontproperties=self.font_prop)
        ax.set_ylabel('重构损失 (MSE)', fontsize=14, fontproperties=self.font_prop)
        ax.set_title('自编码器训练过程', fontsize=16, fontweight='bold', 
                    fontproperties=self.font_prop, pad=20)
        
        ax.legend(fontsize=11, prop=self.font_prop, loc='upper right',
                 framealpha=0.9, edgecolor='gray')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置刻度
        ax.set_xlim(1, len(history))
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"[OK] 训练曲线已保存: {save_path}")
        
        plt.close()
    
    def plot_reconstruction_comparison_enhanced(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        n_samples: int = 5,
        save_path: Optional[str] = None
    ):
        """
        绘制美观的重构效果对比图
        
        Args:
            original: 原始数据
            reconstructed: 重构数据
            n_samples: 展示样本数
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(16, 4*n_samples))
        
        # 使用GridSpec创建灵活布局
        gs = gridspec.GridSpec(n_samples, 3, width_ratios=[4, 4, 2], 
                              hspace=0.3, wspace=0.25)
        
        for i in range(n_samples):
            # 原始数据
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.fill_between(range(len(original[i])), original[i], 
                            alpha=0.4, color=self.COLORS['primary'])
            ax1.plot(original[i], linewidth=1.5, color=self.COLORS['primary'], alpha=0.8)
            ax1.set_title(f'样本 {i+1} - 原始表达谱', fontsize=12, 
                         fontproperties=self.font_prop, fontweight='bold')
            ax1.set_ylabel('表达值', fontsize=10, fontproperties=self.font_prop)
            ax1.grid(True, alpha=0.2)
            ax1.set_xlim(0, len(original[i]))
            
            if i == n_samples - 1:
                ax1.set_xlabel('基因索引', fontsize=10, fontproperties=self.font_prop)
            
            # 重构数据
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.fill_between(range(len(reconstructed[i])), reconstructed[i], 
                            alpha=0.4, color=self.COLORS['accent'])
            ax2.plot(reconstructed[i], linewidth=1.5, color=self.COLORS['accent'], alpha=0.8)
            ax2.set_title(f'样本 {i+1} - 重构表达谱', fontsize=12, 
                         fontproperties=self.font_prop, fontweight='bold')
            ax2.grid(True, alpha=0.2)
            ax2.set_xlim(0, len(reconstructed[i]))
            
            if i == n_samples - 1:
                ax2.set_xlabel('基因索引', fontsize=10, fontproperties=self.font_prop)
            
            # 残差热图
            ax3 = fig.add_subplot(gs[i, 2])
            residual = np.abs(original[i] - reconstructed[i])
            mse = np.mean(residual**2)
            corr = np.corrcoef(original[i], reconstructed[i])[0, 1]
            
            # 绘制残差直方图
            ax3.hist(residual, bins=30, color=self.COLORS['secondary'], 
                    alpha=0.7, edgecolor='white')
            ax3.axvline(np.mean(residual), color='red', linestyle='--', 
                       linewidth=2, label=f'均值={np.mean(residual):.3f}')
            ax3.set_title(f'残差分布\nMSE={mse:.4f}, r={corr:.3f}', 
                         fontsize=10, fontproperties=self.font_prop)
            ax3.legend(fontsize=8, prop=self.font_prop)
            
            if i == n_samples - 1:
                ax3.set_xlabel('残差值', fontsize=9, fontproperties=self.font_prop)
        
        # 总标题
        fig.suptitle('自编码器重构效果对比', fontsize=18, fontweight='bold',
                    fontproperties=self.font_prop, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"[OK] 重构对比图已保存: {save_path}")
        
        plt.close()
    
    def plot_latent_space_analysis_enhanced(
        self,
        latent: np.ndarray,
        labels: Optional[np.ndarray] = None,
        label_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        绘制美观的潜在空间分析图
        
        Args:
            latent: 潜在空间表示
            labels: 样本标签
            label_names: 标签名称
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.3)
        
        # 1. 潜在空间散点图（前两维）
        ax1 = fig.add_subplot(gs[0])
        
        if labels is not None and label_names is not None:
            unique_labels = np.unique(labels)
            for idx, label in enumerate(unique_labels):
                mask = labels == label
                name = label_names[label] if label < len(label_names) else f'类别{label}'
                color = list(self.CELL_COLORS.values())[idx % len(self.CELL_COLORS)]
                ax1.scatter(latent[mask, 0], latent[mask, 1], s=30, alpha=0.6,
                           label=name, c=color, edgecolors='white', linewidths=0.5)
            ax1.legend(fontsize=9, prop=self.font_prop, loc='best', 
                      framealpha=0.9, edgecolor='gray')
        else:
            scatter = ax1.scatter(latent[:, 0], latent[:, 1], s=30, alpha=0.6,
                                 c=np.arange(len(latent)), cmap='viridis',
                                 edgecolors='white', linewidths=0.3)
            plt.colorbar(scatter, ax=ax1, label='样本索引', shrink=0.8)
        
        ax1.set_xlabel('潜在维度 1', fontsize=12, fontproperties=self.font_prop)
        ax1.set_ylabel('潜在维度 2', fontsize=12, fontproperties=self.font_prop)
        ax1.set_title('潜在空间分布', fontsize=14, fontweight='bold',
                     fontproperties=self.font_prop)
        ax1.grid(True, alpha=0.2, linestyle='--')
        
        # 2. 各维度激活强度
        ax2 = fig.add_subplot(gs[1])
        latent_mean = np.abs(latent).mean(axis=0)
        latent_std = np.abs(latent).std(axis=0)
        
        dims = range(len(latent_mean))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(latent_mean)))
        
        bars = ax2.bar(dims, latent_mean, color=colors, edgecolor='white', 
                      linewidth=0.5, alpha=0.9)
        ax2.errorbar(dims, latent_mean, yerr=latent_std, fmt='none', 
                    ecolor='gray', capsize=2, alpha=0.5)
        
        # 标注最高的几个维度
        top_k = 3
        top_dims = np.argsort(latent_mean)[-top_k:]
        for d in top_dims:
            ax2.annotate(f'D{d}', xy=(d, latent_mean[d]), 
                        xytext=(d, latent_mean[d]+0.1),
                        fontsize=8, ha='center', fontproperties=self.font_prop)
        
        ax2.set_xlabel('潜在维度', fontsize=12, fontproperties=self.font_prop)
        ax2.set_ylabel('平均激活值', fontsize=12, fontproperties=self.font_prop)
        ax2.set_title('维度激活强度', fontsize=14, fontweight='bold',
                     fontproperties=self.font_prop)
        ax2.grid(True, alpha=0.2, axis='y')
        
        # 3. 维度相关性热图
        ax3 = fig.add_subplot(gs[2])
        
        # 计算维度间相关性（只取前16维，避免太密集）
        n_show = min(16, latent.shape[1])
        corr_matrix = np.corrcoef(latent[:, :n_show].T)
        
        im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax3, shrink=0.8, label='相关系数')
        
        ax3.set_xlabel('潜在维度', fontsize=12, fontproperties=self.font_prop)
        ax3.set_ylabel('潜在维度', fontsize=12, fontproperties=self.font_prop)
        ax3.set_title(f'前{n_show}维相关性', fontsize=14, fontweight='bold',
                     fontproperties=self.font_prop)
        
        # 设置刻度
        tick_positions = np.arange(0, n_show, max(1, n_show//8))
        ax3.set_xticks(tick_positions)
        ax3.set_yticks(tick_positions)
        
        fig.suptitle('潜在空间综合分析', fontsize=18, fontweight='bold',
                    fontproperties=self.font_prop, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"[OK] 潜在空间分析图已保存: {save_path}")
        
        plt.close()
    
    # ==================== 高级实验可视化 ====================
    
    def plot_performance_comparison_enhanced(
        self,
        results: Dict[str, Dict],
        metrics: List[str] = ['accuracy', 'f1_macro', 'auc'],
        save_path: Optional[str] = None
    ):
        """
        绘制美观的性能对比图
        
        Args:
            results: 各标签比例的结果字典
            metrics: 要展示的指标
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)
        
        # 准备数据
        labels = list(results.keys())
        x = np.arange(len(labels))
        
        # 配色
        metric_colors = {
            'accuracy': self.COLORS['primary'],
            'f1_macro': self.COLORS['accent'],
            'auc': self.COLORS['secondary'],
        }
        
        # 1. 柱状图对比
        ax1 = fig.add_subplot(gs[0, 0])
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [results[label].get(metric, 0) for label in labels]
            offset = width * (i - len(metrics)/2 + 0.5)
            bars = ax1.bar(x + offset, values, width, label=metric.upper(),
                          color=metric_colors.get(metric, f'C{i}'), 
                          edgecolor='white', linewidth=1)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', fontsize=8, fontproperties=self.font_prop)
        
        ax1.set_xlabel('标签比例', fontsize=12, fontproperties=self.font_prop)
        ax1.set_ylabel('性能分数', fontsize=12, fontproperties=self.font_prop)
        ax1.set_title('各指标性能对比', fontsize=14, fontweight='bold',
                     fontproperties=self.font_prop)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontproperties=self.font_prop)
        ax1.legend(fontsize=9, prop=self.font_prop, loc='upper right', 
                   bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
        ax1.set_ylim(0, 1.15)
        ax1.grid(True, alpha=0.2, axis='y')
        
        # 2. 折线图趋势
        ax2 = fig.add_subplot(gs[0, 1])
        
        for metric in metrics:
            values = [results[label].get(metric, 0) for label in labels]
            ax2.plot(labels, values, marker='o', markersize=10, linewidth=2.5,
                    label=metric.upper(), color=metric_colors.get(metric, 'gray'))
            ax2.fill_between(labels, values, alpha=0.1, 
                            color=metric_colors.get(metric, 'gray'))
        
        ax2.set_xlabel('标签比例', fontsize=12, fontproperties=self.font_prop)
        ax2.set_ylabel('性能分数', fontsize=12, fontproperties=self.font_prop)
        ax2.set_title('性能随标签增加的变化趋势', fontsize=14, fontweight='bold',
                     fontproperties=self.font_prop)
        ax2.legend(fontsize=9, prop=self.font_prop, loc='upper left', 
                   bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        ax2.set_ylim(0, 0.7)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 3. 雷达图
        ax3 = fig.add_subplot(gs[1, 0], projection='polar')
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        for i, label in enumerate(labels):
            values = [results[label].get(metric, 0) for metric in metrics]
            values += values[:1]  # 闭合
            ax3.plot(angles, values, 'o-', linewidth=2, label=label, 
                    markersize=6)
            ax3.fill(angles, values, alpha=0.15)
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels([m.upper() for m in metrics], 
                           fontsize=10, fontproperties=self.font_prop)
        ax3.set_ylim(0, 0.7)
        ax3.legend(fontsize=9, prop=self.font_prop, loc='upper left',
                  bbox_to_anchor=(-0.15, 1.15), framealpha=0.9)
        ax3.set_title('多维性能雷达图', fontsize=14, fontweight='bold',
                     fontproperties=self.font_prop, pad=20)
        
        # 4. 效率分析（样本数 vs 性能）
        ax4 = fig.add_subplot(gs[1, 1])
        
        # 假设的样本数量（可以从results中获取实际值）
        sample_counts = {'10%': 255, '30%': 765, '50%': 1275, '100%': 2550}
        
        for metric in metrics[:2]:  # 只画两个主要指标
            x_vals = [sample_counts.get(label, 1000) for label in labels]
            y_vals = [results[label].get(metric, 0) for label in labels]
            
            ax4.scatter(x_vals, y_vals, s=150, label=metric.upper(),
                       color=metric_colors.get(metric, 'gray'),
                       edgecolors='white', linewidths=2, alpha=0.8)
            ax4.plot(x_vals, y_vals, '--', alpha=0.5, 
                    color=metric_colors.get(metric, 'gray'))
        
        ax4.set_xlabel('训练样本数', fontsize=12, fontproperties=self.font_prop)
        ax4.set_ylabel('性能分数', fontsize=12, fontproperties=self.font_prop)
        ax4.set_title('样本效率分析', fontsize=14, fontweight='bold',
                     fontproperties=self.font_prop)
        ax4.legend(fontsize=9, prop=self.font_prop, loc='upper left',
                   bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        fig.suptitle('提示学习模型性能综合分析', fontsize=18, fontweight='bold',
                    fontproperties=self.font_prop, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"[OK] 性能对比图已保存: {save_path}")
        
        plt.close()
    
    def plot_training_history_grid(
        self,
        histories: Dict[str, Dict],
        save_path: Optional[str] = None
    ):
        """
        绘制多个训练历史的网格对比图
        
        Args:
            histories: {标签比例: {train_loss, val_loss, train_acc, val_acc}}
            save_path: 保存路径
        """
        n_configs = len(histories)
        fig, axes = plt.subplots(2, n_configs, figsize=(5*n_configs, 10))
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_configs))
        
        for i, (label, history) in enumerate(histories.items()):
            # 损失曲线
            ax_loss = axes[0, i] if n_configs > 1 else axes[0]
            epochs = range(1, len(history['train_loss']) + 1)
            
            ax_loss.plot(epochs, history['train_loss'], '-', linewidth=2,
                        color=self.COLORS['primary'], label='训练损失', alpha=0.9)
            ax_loss.plot(epochs, history['val_loss'], '--', linewidth=2,
                        color=self.COLORS['accent'], label='验证损失', alpha=0.9)
            ax_loss.fill_between(epochs, history['train_loss'], 
                                alpha=0.2, color=self.COLORS['primary'])
            
            ax_loss.set_title(f'{label}标签 - 损失曲线', fontsize=13, 
                             fontweight='bold', fontproperties=self.font_prop)
            ax_loss.set_xlabel('Epoch', fontsize=11, fontproperties=self.font_prop)
            ax_loss.set_ylabel('Loss', fontsize=11, fontproperties=self.font_prop)
            ax_loss.legend(fontsize=9, prop=self.font_prop)
            ax_loss.grid(True, alpha=0.3)
            
            # 准确率曲线
            ax_acc = axes[1, i] if n_configs > 1 else axes[1]
            
            ax_acc.plot(epochs, history['train_acc'], '-', linewidth=2,
                       color=self.COLORS['secondary'], label='训练准确率', alpha=0.9)
            ax_acc.plot(epochs, history['val_acc'], '--', linewidth=2,
                       color=self.COLORS['success'], label='验证准确率', alpha=0.9)
            ax_acc.fill_between(epochs, history['train_acc'],
                               alpha=0.2, color=self.COLORS['secondary'])
            
            ax_acc.set_title(f'{label}标签 - 准确率曲线', fontsize=13,
                            fontweight='bold', fontproperties=self.font_prop)
            ax_acc.set_xlabel('Epoch', fontsize=11, fontproperties=self.font_prop)
            ax_acc.set_ylabel('Accuracy', fontsize=11, fontproperties=self.font_prop)
            ax_acc.legend(fontsize=9, prop=self.font_prop)
            ax_acc.grid(True, alpha=0.3)
            ax_acc.set_ylim(0, 1.05)
        
        fig.suptitle('不同标签比例下的训练历史对比', fontsize=18, fontweight='bold',
                    fontproperties=self.font_prop, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"[OK] 训练历史网格图已保存: {save_path}")
        
        plt.close()
    
    def plot_parameter_efficiency_chart(
        self,
        total_params: int,
        trainable_params: int,
        frozen_params: int,
        save_path: Optional[str] = None
    ):
        """
        绘制参数效率饼图和对比图
        
        Args:
            total_params: 总参数量
            trainable_params: 可训练参数
            frozen_params: 冻结参数
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 饼图
        ax1 = axes[0]
        sizes = [trainable_params, frozen_params]
        labels_pie = [f'可训练\n{trainable_params:,}\n({100*trainable_params/total_params:.1f}%)',
                     f'冻结\n{frozen_params:,}\n({100*frozen_params/total_params:.1f}%)']
        colors_pie = [self.COLORS['learnable'], self.COLORS['frozen']]
        explode = (0.05, 0)
        
        wedges, texts = ax1.pie(sizes, colors=colors_pie, explode=explode,
                               shadow=True, startangle=90)
        
        # 添加图例
        ax1.legend(wedges, labels_pie, title="参数类型",
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=11, prop=self.font_prop)
        
        ax1.set_title('参数分布', fontsize=14, fontweight='bold',
                     fontproperties=self.font_prop)
        
        # 2. 对比条形图
        ax2 = axes[1]
        categories = ['总参数', '可训练', '冻结']
        values = [total_params, trainable_params, frozen_params]
        colors_bar = ['#3498db', self.COLORS['learnable'], self.COLORS['frozen']]
        
        bars = ax2.barh(categories, values, color=colors_bar, 
                       edgecolor='white', linewidth=2, height=0.6)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax2.text(val + total_params*0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=12, 
                    fontproperties=self.font_prop)
        
        ax2.set_xlabel('参数数量', fontsize=12, fontproperties=self.font_prop)
        ax2.set_title('参数效率对比', fontsize=14, fontweight='bold',
                     fontproperties=self.font_prop)
        ax2.set_xlim(0, total_params * 1.3)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 添加效率指标文本框
        efficiency_text = f"参数效率: 仅需训练 {100*trainable_params/total_params:.1f}% 的参数"
        ax2.text(0.5, -0.15, efficiency_text, transform=ax2.transAxes,
                fontsize=12, fontproperties=self.font_prop,
                ha='center', style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                         edgecolor='orange', alpha=0.8))
        
        fig.suptitle('提示学习参数效率分析', fontsize=18, fontweight='bold',
                    fontproperties=self.font_prop, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"[OK] 参数效率图已保存: {save_path}")
        
        plt.close()
    
    def plot_cross_validation_results(
        self,
        cv_results: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """
        绘制交叉验证结果箱线图
        
        Args:
            cv_results: {指标名: [各折结果]}
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        metrics = list(cv_results.keys())
        data = [cv_results[m] for m in metrics]
        
        # 箱线图
        bp = ax.boxplot(data, patch_artist=True, labels=metrics)
        
        # 设置颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 添加散点
        for i, d in enumerate(data):
            x = np.random.normal(i+1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.6, s=50, c='black', zorder=3)
        
        # 添加均值标记
        means = [np.mean(d) for d in data]
        ax.scatter(range(1, len(metrics)+1), means, marker='D', s=100,
                  c=self.COLORS['accent'], zorder=4, label='均值')
        
        ax.set_ylabel('分数', fontsize=12, fontproperties=self.font_prop)
        ax.set_title('5折交叉验证结果', fontsize=14, fontweight='bold',
                    fontproperties=self.font_prop)
        ax.legend(fontsize=10, prop=self.font_prop)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加均值±标准差标注
        for i, (m, d) in enumerate(zip(metrics, data)):
            mean_val = np.mean(d)
            std_val = np.std(d)
            ax.annotate(f'{mean_val:.3f}±{std_val:.3f}',
                       xy=(i+1, max(d)), xytext=(i+1, max(d)+0.02),
                       ha='center', fontsize=9, fontproperties=self.font_prop)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"[OK] 交叉验证图已保存: {save_path}")
        
        plt.close()


def generate_basic_experiment_figures(
    output_dir: str = 'outputs_basic/plots',
    history: Optional[List[float]] = None,
    original: Optional[np.ndarray] = None,
    reconstructed: Optional[np.ndarray] = None,
    latent: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None
):
    """
    生成初级实验的所有增强可视化图表
    
    Args:
        output_dir: 输出目录
        history: 训练损失历史
        original: 原始数据
        reconstructed: 重构数据
        latent: 潜在空间表示
        labels: 标签
        label_names: 标签名称
    """
    viz = EnhancedVisualizer(output_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if history is not None:
        viz.plot_autoencoder_training_curve_enhanced(
            history, str(output_path / 'training_loss_curve_enhanced.png'))
    
    if original is not None and reconstructed is not None:
        viz.plot_reconstruction_comparison_enhanced(
            original, reconstructed, n_samples=5,
            save_path=str(output_path / 'reconstruction_comparison_enhanced.png'))
    
    if latent is not None:
        viz.plot_latent_space_analysis_enhanced(
            latent, labels, label_names,
            save_path=str(output_path / 'latent_space_analysis_enhanced.png'))


def generate_advanced_experiment_figures(
    output_dir: str = 'outputs_advanced/plots',
    results: Optional[Dict] = None,
    histories: Optional[Dict] = None,
    total_params: int = 737989,
    trainable_params: int = 183813,
    frozen_params: int = 554176,
    cv_results: Optional[Dict] = None
):
    """
    生成高级实验的所有增强可视化图表
    
    Args:
        output_dir: 输出目录
        results: 性能结果
        histories: 训练历史
        total_params: 总参数
        trainable_params: 可训练参数
        frozen_params: 冻结参数
        cv_results: 交叉验证结果
    """
    viz = EnhancedVisualizer(output_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if results is not None:
        viz.plot_performance_comparison_enhanced(
            results, save_path=str(output_path / 'performance_comparison_enhanced.png'))
    
    if histories is not None:
        viz.plot_training_history_grid(
            histories, save_path=str(output_path / 'training_history_grid.png'))
    
    viz.plot_parameter_efficiency_chart(
        total_params, trainable_params, frozen_params,
        save_path=str(output_path / 'parameter_efficiency.png'))
    
    if cv_results is not None:
        viz.plot_cross_validation_results(
            cv_results, save_path=str(output_path / 'cross_validation_results.png'))


if __name__ == '__main__':
    # 测试代码
    print("测试增强可视化模块...")
    
    # 模拟数据
    np.random.seed(42)
    
    # 初级实验测试
    history = list(np.exp(-np.linspace(0, 3, 100)) * 0.5 + 0.7 + np.random.randn(100)*0.02)
    original = np.random.randn(5, 200)
    reconstructed = original + np.random.randn(5, 200) * 0.1
    latent = np.random.randn(500, 64)
    
    generate_basic_experiment_figures(
        output_dir='test_outputs/basic',
        history=history,
        original=original,
        reconstructed=reconstructed,
        latent=latent
    )
    
    # 高级实验测试
    results = {
        '10%': {'accuracy': 0.78, 'f1_macro': 0.75, 'auc': 0.85},
        '30%': {'accuracy': 0.82, 'f1_macro': 0.79, 'auc': 0.88},
        '50%': {'accuracy': 0.85, 'f1_macro': 0.82, 'auc': 0.90},
        '100%': {'accuracy': 0.88, 'f1_macro': 0.85, 'auc': 0.92},
    }
    
    generate_advanced_experiment_figures(
        output_dir='test_outputs/advanced',
        results=results
    )
    
    print("测试完成！")
