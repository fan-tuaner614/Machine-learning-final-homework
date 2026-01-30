"""
步骤1: 数据加载模块
Data Loader Module for Single-Cell RNA-seq Data
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SingleCellDataLoader:
    """单细胞RNA-seq数据加载器"""
    
    def __init__(self, data_dir: str = 'data'):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load_all_data(self) -> Dict[str, sc.AnnData]:
        """
        加载所有数据文件
        
        Returns:
            包含所有数据集的字典
        """
        print("="*60)
        print("步骤1: 数据加载")
        print("="*60)
        
        # 加载h5ad格式的单细胞数据
        for file_path in self.data_dir.glob('*.h5ad'):
            try:
                adata = sc.read_h5ad(file_path)
                dataset_name = file_path.stem
                self.datasets[dataset_name] = adata
                
                print(f"\n[OK] 成功加载: {file_path.name}")
                print(f"  细胞数量: {adata.n_obs}")
                print(f"  基因数量: {adata.n_vars}")
                print(f"  观测属性: {list(adata.obs.columns)}")
                
                # 显示细胞亚型分布
                if 'subtype' in adata.obs.columns:
                    subtype_counts = adata.obs['subtype'].value_counts()
                    print(f"  细胞亚型分布:")
                    for subtype, count in subtype_counts.items():
                        print(f"    - {subtype}: {count} 个细胞 ({count/adata.n_obs*100:.1f}%)")
                        
            except Exception as e:
                print(f"\n[ERROR] 加载 {file_path.name} 失败: {e}")
        
        # 加载.npy格式的数据（如果存在）
        for file_path in self.data_dir.glob('*.npy'):
            try:
                data = np.load(file_path, allow_pickle=True)
                dataset_name = file_path.stem
                self.datasets[f"{dataset_name}_npy"] = data
                print(f"\n[OK] 成功加载: {file_path.name}")
                print(f"  数据形状: {data.shape}")
            except Exception as e:
                print(f"\n⚠ 跳过 {file_path.name}: {e}")
        
        print(f"\n{'='*60}")
        print(f"数据加载完成！共加载 {len(self.datasets)} 个数据集")
        print(f"{'='*60}\n")
        
        return self.datasets
    
    def get_dataset(self, name: str) -> Optional[sc.AnnData]:
        """
        获取指定的数据集
        
        Args:
            name: 数据集名称
            
        Returns:
            AnnData对象或None
        """
        return self.datasets.get(name)
    
    def get_all_datasets(self) -> Dict[str, sc.AnnData]:
        """获取所有数据集"""
        return self.datasets
    
    def get_dataset_info(self, name: str) -> Dict:
        """
        获取数据集的详细信息
        
        Args:
            name: 数据集名称
            
        Returns:
            包含数据集信息的字典
        """
        if name not in self.datasets:
            return {}
        
        adata = self.datasets[name]
        
        info = {
            'name': name,
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'obs_keys': list(adata.obs.columns),
            'var_keys': list(adata.var.columns) if len(adata.var.columns) > 0 else [],
        }
        
        # 添加细胞亚型信息
        if 'subtype' in adata.obs.columns:
            info['subtypes'] = adata.obs['subtype'].value_counts().to_dict()
            info['n_subtypes'] = adata.obs['subtype'].nunique()
        
        return info
    
    def split_by_subtype(self, dataset_name: str) -> Dict[str, sc.AnnData]:
        """
        按细胞亚型分割数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            按亚型分割的数据集字典
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集 {dataset_name} 不存在")
        
        adata = self.datasets[dataset_name]
        
        if 'subtype' not in adata.obs.columns:
            raise ValueError(f"数据集 {dataset_name} 没有 'subtype' 列")
        
        split_data = {}
        for subtype in adata.obs['subtype'].unique():
            mask = adata.obs['subtype'] == subtype
            split_data[subtype] = adata[mask].copy()
        
        return split_data
    
    def get_train_test_datasets(self) -> Tuple[list, list]:
        """
        获取训练集和测试集列表
        
        Returns:
            (训练集列表, 测试集列表)
        """
        train_datasets = [name for name in self.datasets.keys() 
                         if 'SC-1' in name or 'train' in name.lower()]
        test_datasets = [name for name in self.datasets.keys() 
                        if 'SC-2' in name or 'test' in name.lower()]
        
        return train_datasets, test_datasets
    
    def load_labeled_subset(self, dataset_name: str, indices_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据索引文件从数据集中加载有标签的子集
        
        Args:
            dataset_name: 数据集名称（如 'SC-2_dense'）
            indices_file: 索引文件路径（如 'data/10_percent.npy'）
        
        Returns:
            (data, labels, indices): 数据、标签和索引
        """
        # 加载数据集
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集 {dataset_name} 不存在，请先调用 load_all_data()")
        
        adata = self.datasets[dataset_name]
        
        # 加载索引
        indices_path = Path(indices_file)
        if not indices_path.exists():
            raise FileNotFoundError(f"索引文件 {indices_file} 不存在")
        
        indices = np.load(indices_file, allow_pickle=True)
        
        print(f"\n加载有标签子集:")
        print(f"  数据集: {dataset_name}")
        print(f"  索引文件: {indices_path.name}")
        print(f"  索引数量: {len(indices)}")
        
        # 提取对应索引的数据
        # 注意：索引可能是布尔数组或整数索引
        if indices.dtype == bool:
            subset_adata = adata[indices].copy()
        else:
            subset_adata = adata[indices.astype(int)].copy()
        
        # 获取表达矩阵
        if hasattr(subset_adata.X, 'toarray'):
            data = subset_adata.X.toarray()
        else:
            data = subset_adata.X
        
        # 获取标签
        if 'subtype' in subset_adata.obs.columns:
            labels = subset_adata.obs['subtype'].values
            
            # 显示标签分布
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"  标签分布:")
            for label, count in zip(unique_labels, counts):
                print(f"    - {label}: {count} ({count/len(labels)*100:.1f}%)")
        else:
            raise ValueError(f"数据集 {dataset_name} 没有 'subtype' 列")
        
        return data, labels, indices
    
    def load_all_labeled_subsets(self, dataset_name: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        加载所有标签比例的子集
        
        Args:
            dataset_name: 数据集名称（如 'SC-2_dense'）
        
        Returns:
            字典，键为标签比例，值为 (data, labels, indices)
        """
        subset_files = ['10_percent.npy', '30_percent.npy', '50_percent.npy', 'all.npy']
        subsets = {}
        
        print(f"\n{'='*60}")
        print(f"加载所有标签比例的子集")
        print(f"{'='*60}")
        
        for subset_file in subset_files:
            subset_path = self.data_dir / subset_file
            if subset_path.exists():
                try:
                    subset_name = subset_file.replace('.npy', '').replace('_', ' ').title()
                    data, labels, indices = self.load_labeled_subset(dataset_name, str(subset_path))
                    subsets[subset_file.replace('.npy', '')] = (data, labels, indices)
                except Exception as e:
                    print(f"  ⚠ 加载 {subset_file} 失败: {e}")
        
        print(f"\n{'='*60}")
        print(f"成功加载 {len(subsets)} 个子集")
        print(f"{'='*60}\n")
        
        return subsets


def main():
    """测试数据加载模块"""
    loader = SingleCellDataLoader(data_dir='data')
    datasets = loader.load_all_data()
    
    # 显示每个数据集的详细信息
    for name in datasets.keys():
        if isinstance(datasets[name], sc.AnnData):
            info = loader.get_dataset_info(name)
            print(f"\n数据集 '{name}' 详细信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
