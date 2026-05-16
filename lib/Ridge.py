import numpy as np
import torch
import warnings


class Ridge_Regression:
    def __init__(self, X, y, alpha, fit_intercept=False, device='cuda'):
        # Keep data on GPU using PyTorch instead of CuPy
        if hasattr(X, 'cpu'):
            self.X = X.float().to(device)  # 确保张量在正确设备上
        else:
            self.X = torch.tensor(X, dtype=torch.float32, device=device)
            
        if hasattr(y, 'cpu'):
            self.y = y.float().to(device)  # 确保张量在正确设备上
        else:
            self.y = torch.tensor(y, dtype=torch.float32, device=device)
            
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.n_samples, self.n_features = X.shape

    def fit(self):
        coef = self._solve_cholesky()
        return coef

    def _solve_cholesky(self):
        n_samples = self.X.shape[1]
        n_targets = self.y.shape[0]
        
        # 一次性计算，减少重复操作
        K = self.safe_sparse_dot(self.X.T, self.X, dense_output=True)
        Xy = self.safe_sparse_dot(self.X.T, self.y, dense_output=True)

        # Add regularization to diagonal - PyTorch version
        diag_indices = torch.arange(min(K.shape), device=K.device)
        K[diag_indices, diag_indices] += self.alpha

        # Use PyTorch's linear algebra - 优化内存使用
        try:
            solution = torch.linalg.solve(K, Xy)
        except torch.linalg.LinAlgError:
            # 如果求解失败，使用伪逆
            solution = torch.linalg.pinv(K) @ Xy
        
        if solution.ndim == 1:
            # For 1D tensors, just return as is
            return solution
        elif solution.ndim == 2:
            # For 2D tensors, transpose normally
            return solution.T
        else:
            # For higher dimensions, transpose the last two dimensions
            return solution.transpose(-2, -1)

    def safe_sparse_dot(self, a, b, *, dense_output=False):
        """Dot product that handle the sparse matrix case correctly.
        Parameters
        ----------
        a : {ndarray, sparse matrix}
        b : {ndarray, sparse matrix}
        dense_output : bool, default=False
            When False, ``a`` and ``b`` both being sparse will yield sparse output.
            When True, output will always be a dense array.
        Returns
        -------
        dot_product : {ndarray, sparse matrix}
            Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
        """
        # Simplified version for dense PyTorch tensors
        if a.ndim > 2 or b.ndim > 2:
            # For 3D+ tensors, use batched matrix multiplication
            ret = torch.matmul(a, b)
        else:
            ret = a @ b

        return ret


