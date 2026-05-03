import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import warnings

def optimize_F_gradient_descent_local_large_scale(A, B, F, Q, R, w, X_0, mask, gam, dzm):
    # 专为大尺度系统设计的梯度下降优化器 (规避 kron 张量积内存爆炸)
    
    n = A.shape[0]
    
    # --- 0. 强制稀疏化数据结构以启用底层加速 ---
    A = sparse.csr_matrix(A)
    B = sparse.csr_matrix(B)
    F = sparse.csr_matrix(F)
    Q = sparse.csr_matrix(Q)
    R = sparse.csr_matrix(R)
    X_0 = sparse.csr_matrix(X_0)
    mask = sparse.csr_matrix(mask)
    
    # 预存恒定转置矩阵
    A_T = A.transpose()
    
    for ii in range(1, dzm + 1):
        A_cl = A + B @ F
        A_cl_T = A_cl.transpose()
        
        # 1. 迭代求解代价矩阵 G (后向可观性方向)
        Q_term_G = Q + F.transpose() @ R @ F
        G = solve_generalized_lyap_iter_sparse(A_cl, A_cl_T, A, A_T, w, Q_term_G, 'G')
        
        # 2. 迭代求解状态协方差矩阵 Y (前向可控性方向)
        Y = solve_generalized_lyap_iter_sparse(A_cl, A_cl_T, A, A_T, w, X_0, 'Y')
        
        # 3. 计算梯度
        D_J = B.transpose() @ G @ A_cl @ Y + R @ F @ Y
        
        # 4. 施加网络拓扑掩码约束 (Hadamard 乘积)
        if sparse.issparse(D_J):
            D_J = D_J.multiply(mask)
        else:
            D_J = D_J * mask.toarray()
        
        # 5. 归一化梯度下降更新
        if sparse.issparse(D_J):
            grad_norm = spla.norm(D_J, 'fro')
        else:
            grad_norm = np.linalg.norm(D_J, 'fro')

        F = F - gam * (D_J / grad_norm)
        
        # 监控迭代过程
        if ii % 10 == 0:
            print(f"Iter: {ii}, Grad Norm: {grad_norm}")
            
    # 最终结果结算
    F_opt = F.toarray() if sparse.issparse(F) else np.array(F)
    
    A_cl_opt = A + B @ F
    Q_term_G_final = Q + F.transpose() @ R @ F
    G_final = solve_generalized_lyap_iter_sparse(A_cl_opt, A_cl_opt.transpose(), A, A_T, w, Q_term_G_final, 'G')
    
    cost_matrix = G_final @ X_0
    J_opt = cost_matrix.diagonal().sum() if sparse.issparse(cost_matrix) else np.trace(cost_matrix)
    
    return F_opt, J_opt


def solve_generalized_lyap_iter_sparse(A_cl, A_cl_T, A, A_T, w, Q_term, mode):
    # 迭代法求解 Lyapunov 方程，规避高维矩阵左除
    tol = 1e-6
    max_iter = 10000
    
    # 以常数项作为迭代基态
    P = Q_term 
    
    for iter_num in range(1, max_iter + 1):
        if mode == 'G':
            P_new = A_cl_T @ P @ A_cl + w * (A_T @ P @ A) + Q_term
        else:
            P_new = A_cl @ P @ A_cl_T + w * (A @ P @ A_T) + Q_term
            
        diff = P_new - P
        if sparse.issparse(diff):
            err = spla.norm(diff, 'fro') / (spla.norm(P, 'fro') + np.finfo(float).eps)
        else:
            err = np.linalg.norm(diff, 'fro') / (np.linalg.norm(P, 'fro') + np.finfo(float).eps)
            
        P = P_new
        
        if err < tol:
            break
            
    if iter_num == max_iter:
        warnings.warn(f"Lyapunov迭代达到最大次数 {max_iter}，最终残差: {err:e}。系统可能濒临失稳。")
        
    return P