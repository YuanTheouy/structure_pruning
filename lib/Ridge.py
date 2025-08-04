import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import lsmr
import warnings


class Ridge_Regression:
    def __init__(self, X, y, alpha, fit_intercept=False):
        self.X = cp.asarray(X.cpu().numpy(), dtype=cp.float32)
        self.y = cp.asarray(y.cpu().numpy(), dtype=cp.float32)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.n_samples, self.n_features = X.shape
        # self.n_samples, self.n_targets = y.shape

    def fit(self):
        coef = self._solve_cholesky()
        return coef

    def _solve_cholesky(self):
    
        n_samples = self.X.shape[1]
        n_targets = self.y.shape[0]
        
        K = self.safe_sparse_dot(self.X.T, self.X, dense_output=True)
        Xy = self.safe_sparse_dot(self.X.T, self.y, dense_output=True)

        K.flat[::n_samples + 1] += self.alpha #* (self.gamma_1/self.sigma_1)**2 * self.alpha_scale_list @ self.alpha_scale_list.T + self.alpha[1]*self.k_scale  # -- 22 without considering sig,gam

        # try:
        #     dual_coef = cp.linalg.solve(K, self.y)
        # except cp.linalg.LinAlgError:
        #     warnings.warn(
        #         "Singular matrix in solving dual problem. Using "
        #         "least-squares solution instead."
        #     )
        #     dual_coef = cp.linalg.lstsq(K, self.y)[0]
        #
        # coef = self.safe_sparse_dot(self.X.T, dual_coef, dense_output=True).T

        return cp.linalg.solve(K, Xy).T
        # return lsmr(K, Xy)[0].T

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
        if a.ndim > 2 or b.ndim > 2:
            if cp.sparse.issparse(a):  # True if x is a sparse matrix, False otherwise
                # sparse is always 2D. Implies b is 3D+
                # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
                b_ = cp.rollaxis(b, -2)
                b_2d = b_.reshape((b.shape[-2], -1))
                ret = a @ b_2d
                ret = ret.reshape(a.shape[0], *b_.shape[1:])
            elif cp.sparse.issparse(b):
                # sparse is always 2D. Implies a is 3D+
                # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
                a_2d = a.reshape(-1, a.shape[-1])
                ret = a_2d @ b
                ret = ret.reshape(*a.shape[:-1], b.shape[1])
            else:
                ret = cp.dot(a, b)
        else:
            ret = a @ b

        if (cp.sparse.issparse(a) and cp.sparse.issparse(b)
                and dense_output and hasattr(ret, "toarray")):
            return ret.toarray()
        return ret


