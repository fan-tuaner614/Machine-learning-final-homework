"""
步骤2: 数据预处理模块
Preprocessing Module for Single-Cell RNA-seq Data
"""

import numpy as np
import scanpy as sc
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SingleCellPreprocessor:
    """单细胞数据预处理器"""
    
    def __init__(self, 
                 min_genes: int = 200,
                 min_cells: int = 3,
                 n_top_genes: int = 2000,
                 n_pca_components: int = 50):
        """
        初始化预处理器
        
        Args:
            min_genes: 每个细胞最少基因数
            min_cells: 每个基因最少细胞数
            n_top_genes: 高变基因数量
            n_pca_components: PCA主成分数量
        """
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.n_top_genes = n_top_genes
        self.n_pca_components = n_pca_components
        
    def preprocess(self, adata: sc.AnnData, 
                   dataset_name: str = "dataset") -> sc.AnnData:
        """
        完整的预处理流程
        
        Args:
            adata: AnnData对象
            dataset_name: 数据集名称
            
        Returns:
            预处理后的AnnData对象
        """
        print("="*60)
        print(f"步骤2: 数据预处理 - {dataset_name}")
        print("="*60)
        
        # 复制数据避免修改原始数据
        adata = adata.copy()
        
        # 2.1 保存原始计数
        print("\n[2.1] 保存原始计数...")
        if 'counts' not in adata.layers:
            adata.layers['counts'] = adata.X.copy()
        
        # 2.2 质量控制
        print("\n[2.2] 质量控制...")
        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars
        
        sc.pp.filter_cells(adata, min_genes=self.min_genes)
        sc.pp.filter_genes(adata, min_cells=self.min_cells)
        
        print(f"  细胞: {n_cells_before} → {adata.n_obs} "
              f"(保留 {adata.n_obs/n_cells_before*100:.1f}%)")
        print(f"  基因: {n_genes_before} → {adata.n_vars} "
              f"(保留 {adata.n_vars/n_genes_before*100:.1f}%)")
        
        # 2.3 归一化
        print("\n[2.3] 归一化 (每个细胞总计数归一化到10,000)...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        
        # 2.4 对数转换
        print("\n[2.4] 对数转换 log1p...")
        sc.pp.log1p(adata)
        
        # 2.5 识别高变基因
        print(f"\n[2.5] 识别高变基因 (选择 {self.n_top_genes} 个)...")
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=self.n_top_genes,
                flavor='seurat_v3',
                layer='counts'
            )
            
            # 只保留高变基因
            n_hvg = adata.var['highly_variable'].sum()
            adata = adata[:, adata.var.highly_variable].copy()
            print(f"  [OK] 使用 {n_hvg} 个高变基因")
            
        except Exception as e:
            print(f"  [WARNING] HVG识别失败: {e}")
            print(f"  使用前 {self.n_top_genes} 个基因")
            # 确保使用有效的基因索引
            n_genes_to_use = min(self.n_top_genes, adata.n_vars)
            adata = adata[:, :n_genes_to_use].copy()
            
        # 检查并修复稀疏矩阵问题
        if hasattr(adata.X, 'toarray'):
            print(f"  转换稀疏矩阵为密集矩阵...")
            import scipy.sparse as sp
            adata.X = adata.X.toarray()
            print(f"  [OK] 矩阵转换完成")
        
        # 2.6 标准化 - 跳过！
        # 对于单细胞RNA-seq数据，log1p转换后直接PCA是最佳实践
        # StandardScaler会因为稀疏数据产生大量NaN
        print("\n[2.6] 跳过标准化步骤（单细胞最佳实践：log1p后直接PCA）...")
        # 注释掉标准化步骤，避免NaN问题
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(adata.X)
        # adata.X = X_scaled
        
        # 2.7 处理异常值（检查log1p后是否有异常值）
        print("\n[2.7] 检查异常值 (NaN, Inf)...")
        nan_count = np.isnan(adata.X).sum()
        inf_count = np.isinf(adata.X).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  发现 {nan_count} 个NaN, {inf_count} 个Inf")
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=3.0, neginf=-3.0)
            print("  [OK] 已处理异常值")
        else:
            print("  [OK] 无异常值")
        
        # 2.8 PCA降维
        print(f"\n[2.8] PCA降维 (保留 {self.n_pca_components} 个主成分)...")
        n_comps = min(self.n_pca_components, adata.n_obs - 1, adata.n_vars)
        sc.tl.pca(adata, n_comps=n_comps, svd_solver='arpack')
        
        # 计算方差解释比例
        if 'pca' in adata.uns and 'variance_ratio' in adata.uns['pca']:
            variance_ratio = adata.uns['pca']['variance_ratio']
            cumsum_variance = np.cumsum(variance_ratio)
            n_show = min(10, len(cumsum_variance))
            print(f"  前{n_show}个PC解释方差: {cumsum_variance[n_show-1]:.2%}")
            if len(cumsum_variance) >= 20:
                print(f"  前20个PC解释方差: {cumsum_variance[19]:.2%}")
            print(f"  所有PC解释方差: {cumsum_variance[-1]:.2%}")
        
        print(f"\n{'='*60}")
        print(f"预处理完成！")
        print(f"  最终细胞数: {adata.n_obs}")
        print(f"  最终基因数: {adata.n_vars}")
        print(f"  PCA维度: {self.n_pca_components}")
        print(f"{'='*60}\n")
        
        return adata
    
    def prepare_features_labels(self, 
                                adata: sc.AnnData,
                                use_pca: bool = True,
                                n_components: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        准备机器学习的特征和标签
        
        Args:
            adata: 预处理后的AnnData对象
            use_pca: 是否使用PCA特征
            n_components: 使用的主成分数量（None表示使用全部）
            
        Returns:
            (特征矩阵, 标签数组, 标签映射字典)
        """
        # 提取特征
        if use_pca and 'X_pca' in adata.obsm:
            if n_components is None:
                X = adata.obsm['X_pca']
            else:
                X = adata.obsm['X_pca'][:, :n_components]
            print(f"使用PCA特征: {X.shape[1]} 维")
        else:
            X = adata.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            print(f"使用原始特征: {X.shape[1]} 维")
        
        # 提取标签
        if 'subtype' not in adata.obs.columns:
            raise ValueError("数据集没有 'subtype' 列")
        
        labels_str = adata.obs['subtype'].values
        
        # 创建标签映射
        unique_labels = sorted(set(labels_str))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        # 转换为数字标签
        y = np.array([label_to_idx[label] for label in labels_str])
        
        print(f"标签统计:")
        for label, idx in label_to_idx.items():
            count = np.sum(y == idx)
            print(f"  {label}: {count} 个样本 ({count/len(y)*100:.1f}%)")
        
        label_mapping = {
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'n_classes': len(unique_labels)
        }
        
        return X, y, label_mapping
    
    def compute_gene_statistics(self, adata: sc.AnnData) -> dict:
        """
        计算基因统计信息
        
        Args:
            adata: AnnData对象
            
        Returns:
            统计信息字典
        """
        stats = {
            'mean_expression': np.mean(adata.X, axis=0),
            'std_expression': np.std(adata.X, axis=0),
            'cv': np.std(adata.X, axis=0) / (np.mean(adata.X, axis=0) + 1e-10)
        }
        
        return stats


def main():
    """测试预处理模块"""
    from data_loader import SingleCellDataLoader
    
    # 加载数据
    loader = SingleCellDataLoader(data_dir='data')
    datasets = loader.load_all_data()
    
    # 预处理第一个数据集
    if datasets:
        first_dataset_name = list(datasets.keys())[0]
        adata = datasets[first_dataset_name]
        
        if isinstance(adata, sc.AnnData):
            # 初始化预处理器
            preprocessor = SingleCellPreprocessor(
                min_genes=200,
                min_cells=3,
                n_top_genes=2000,
                n_pca_components=50
            )
            
            # 预处理
            adata_processed = preprocessor.preprocess(adata, first_dataset_name)
            
            # 准备特征和标签
            if 'subtype' in adata_processed.obs.columns:
                X, y, label_mapping = preprocessor.prepare_features_labels(
                    adata_processed, 
                    use_pca=True
                )
                print(f"\n特征矩阵形状: {X.shape}")
                print(f"标签数组形状: {y.shape}")
                print(f"类别数量: {label_mapping['n_classes']}")


if __name__ == '__main__':
    main()
