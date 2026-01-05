#!/usr/bin/env python3
"""
Utility Functions
"""

import torch
import numpy as np
import torch

def generate_shifted_matrix(n, device=None):
    
    matrix_columns = []
    granularities = [2, 4]
    
    for granularity in granularities:
        if granularity > n:
            continue
            
        # Calculate step size for this granularity
        step_size = max(1, granularity // 2)
        max_start = n - granularity
        
        for start in range(0, max_start + 1, step_size):
            column = torch.zeros(n, dtype=torch.int, device=device)
            column[start:start + granularity] = 1
            matrix_columns.append(column)
        
        # If the last position is not covered, add a mask at the end
        if max_start >= 0 and (max_start % step_size) != 0:
            column = torch.zeros(n, dtype=torch.int, device=device)
            column[-granularity:] = 1
            matrix_columns.append(column)
    
    if not matrix_columns:
        column = torch.ones(n, dtype=torch.int, device=device)
        matrix_columns.append(column)

    result = torch.stack(matrix_columns, dim=1).unsqueeze(0).expand(1, -1, -1)
    return result

def create_attention_mask(shift_matrix: torch.Tensor) -> torch.Tensor:
    """
    Create attention mask from shift matrix
    
    Args:
        shift_matrix (torch.Tensor): shift matrix, shape [num_chunks, seq_len]
        
    Returns:
        torch.Tensor: attention mask, shape [1, num_chunks, seq_len, seq_len]
    """
    # Transpose and create attention mask
    attention_mask = shift_matrix.transpose(0, 1)  # [seq_len, num_chunks]
    attention_mask = torch.where(attention_mask == 1.0, 0.0, float('-inf'))
    
    # Add dimensions to match expected shape of attention
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, num_chunks]
    
    return attention_mask

def normalize_embeddings(embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    L2 normalize embeddings
    
    Args:
        embeddings (torch.Tensor): Embeddings
        eps (float): Small value to prevent division by zero
        
    Returns:
        torch.Tensor: Normalized embeddings
    """
    norm = torch.norm(embeddings, dim=-1, keepdim=True)
    return embeddings / (norm + eps)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculate cosine similarity
    
    Args:
        a (torch.Tensor): Vector A
        b (torch.Tensor): Vector B
        
    Returns:
        torch.Tensor: Cosine similarity
    """
    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return torch.sum(a_norm * b_norm, dim=-1)

def batch_cosine_similarity(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
    """
    Calculate batch cosine similarity
    
    Args:
        embeddings1 (torch.Tensor): Embeddings group 1, shape [N, dim]
        embeddings2 (torch.Tensor): Embeddings group 2, shape [M, dim]
        
    Returns:
        torch.Tensor: Similarity matrix, shape [N, M]
    """
    embeddings1_norm = normalize_embeddings(embeddings1)
    embeddings2_norm = normalize_embeddings(embeddings2)
    
    return torch.matmul(embeddings1_norm, embeddings2_norm.transpose(0, 1))

def split_embeddings_by_shift_matrix(embeddings: torch.Tensor, shift_matrix: torch.Tensor) -> list:
    """
    Split embeddings based on shift matrix
    
    Args:
        embeddings (torch.Tensor): Embeddings, shape [seq_len, hidden_dim]
        shift_matrix (torch.Tensor): shift matrix, shape [num_chunks, seq_len]
        
    Returns:
        list: List of split embeddings
    """
    split_embeddings = []
    num_chunks, seq_len = shift_matrix.shape
    
    for chunk_idx in range(num_chunks):
        mask = shift_matrix[chunk_idx]  # [seq_len]
        indices = torch.nonzero(mask, as_tuple=True)[0]  # Get indices of non-zero positions
        
        if len(indices) > 0:
            chunk_embeddings = embeddings[indices]  # [chunk_size, hidden_dim]
            split_embeddings.append(chunk_embeddings)
    
    return split_embeddings

def pool_embeddings(embeddings: torch.Tensor, method: str = 'mean') -> torch.Tensor:
    """
    Pool embeddings
    
    Args:
        embeddings (torch.Tensor): Embeddings, shape [seq_len, hidden_dim]
        method (str): Pooling method, optional 'mean', 'max', 'first', 'last'
        
    Returns:
        torch.Tensor: Pooled vector, shape [hidden_dim]
    """
    if method == 'mean':
        return torch.mean(embeddings, dim=0)
    elif method == 'max':
        return torch.max(embeddings, dim=0)[0]
    elif method == 'first':
        return embeddings[0]
    elif method == 'last':
        return embeddings[-1]
    else:
        raise ValueError(f"Unknown pooling method: {method}")

def aggregate_chunk_embeddings(split_embeddings: list, method: str = 'mean') -> torch.Tensor:
    """
    Aggregate chunk embeddings
    
    Args:
        split_embeddings (list): List of split embeddings
        method (str): Aggregation method
        
    Returns:
        torch.Tensor: Aggregated embeddings, shape [num_chunks, hidden_dim]
    """
    if not split_embeddings:
        return torch.tensor([])
    
    aggregated = []
    for chunk_embeddings in split_embeddings:
        pooled = pool_embeddings(chunk_embeddings, method)
        aggregated.append(pooled)
    
    return torch.stack(aggregated)

def safe_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert tensor to numpy array
    
    Args:
        tensor (torch.Tensor): Input tensor
        
    Returns:
        np.ndarray: Numpy array
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    return tensor.numpy()

def ensure_tensor_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Ensure tensor is on specified device
    
    Args:
        tensor (torch.Tensor): Input tensor
        device (torch.device): Target device
        
    Returns:
        torch.Tensor: Tensor on target device
    """
    if tensor.device != device:
        tensor = tensor.to(device)
    return tensor

def get_available_device() -> torch.device:
    """
    Get available device
    
    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def print_tensor_info(tensor: torch.Tensor, name: str = "tensor"):
    """
    Print tensor info
    
    Args:
        tensor (torch.Tensor): Input tensor
        name (str): Tensor name
    """
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Data Type: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Requires Grad: {tensor.requires_grad}")
    if tensor.numel() > 0:
        print(f"  Value Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
        print(f"  Mean: {tensor.mean().item():.6f}")
        print(f"  Std Dev: {tensor.std().item():.6f}")
