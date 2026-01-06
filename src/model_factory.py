import torch
import torch.nn as nn
from tabm import TabM
from rtdl_num_embeddings import PiecewiseLinearEmbeddings

# src/model_factory.py

def create_tabm_model(X_train_sample, device='cuda'):
    """
    X_train_sample: 传入训练集的 Tensor，用于计算 PLE 的分桶统计量
    """
    n_num_features = X_train_sample.shape[1]
    
    # 计算分段线性嵌入的统计信息 (这是 PLE 的核心步骤)
    num_embeddings = PiecewiseLinearEmbeddings.compute_stats(
        X_train_sample.cpu(), 
        n_bins=48
    )
    
    model = TabM.make(
        n_num_features=n_num_features,
        d_out=1,
        num_embeddings=num_embeddings,
        arch_type='tabm',
        k=32,
        # ... 其他参数
    )
    
    return model.to(device)

def compute_tabm_loss(y_pred, y_true):
    k = y_pred.shape[1]
    y_true_stacked = y_true.unsqueeze(1).expand(-1, k, -1)
    # 使用 MSE 预测 Return 的数值
    return nn.MSELoss()(y_pred, y_true_stacked)