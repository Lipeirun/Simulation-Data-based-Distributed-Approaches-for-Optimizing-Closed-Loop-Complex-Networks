import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
from build_noise_model import build_noise_model # 依赖于外部函数构建噪声

def optimize_F_trajectory_local_large_scale(A, B, F0, mask, gam, dzm, T, D, Q, R, w, numAgents, ex_w, W, X_0):
    # 专为大规模系统设计的轨迹采样优化器F_opt
    
    n = A.shape[0]
    m = B.shape[1]
    
    # 0. 强制稀疏化预处理
    A = sparse.csr_matrix(A)
    B = sparse.csr_matrix(B)
    F = sparse.csr_matrix(F0)
    Q = sparse.csr_matrix(Q)
    R = sparse.csr_matrix(R)
    mask = sparse.csr_matrix(mask)
    X_0 = sparse.csr_matrix(X_0)
    A_T = A.transpose()
    
    y_bar_init = np.zeros((n, 1))
    zeta_term = np.zeros((n, T + 1))
    J_record = np.zeros((dzm, 1))
    
    for ii in range(1, dzm + 1):
        # 1. 公式 28：物理稳态轨迹生成
        y_bar = np.zeros((n, D + T + 1))
        y_bar[:, 0:1] = y_bar_init
        
        for k_math in range(D + T):
            nu_k = np.random.normal(0, np.sqrt(w))
            E_noise, _ = build_noise_model(numAgents, ex_w, W)
            sys_mat = A + B @ F + nu_k * A
            y_bar[:, k_math+1:k_math+2] = sys_mat @ y_bar[:, k_math:k_math+1] + E_noise
            
        # 2. 公式 29/30：轨迹采样阶段
        E_sy_sum = np.zeros((m, n))
        E_zy_sum = np.zeros((m, n))
        
        for mu in range(T + 1):
            nu_traj = np.random.normal(0, np.sqrt(w), T)
            y_traj = np.zeros((n, T + 1))
            y_traj[:, 0:1] = y_bar[:, D + mu : D + mu + 1]
            
            y_traj[:, 1:2] = (A + B @ F) @ y_traj[:, 0:1]
            for k_math in range(1, T):
                sys_mat_traj = A + B @ F + nu_traj[k_math - 1] * A
                y_traj[:, k_math+1:k_math+2] = sys_mat_traj @ y_traj[:, k_math:k_math+1]
                
            zeta_traj = np.zeros((n, T + 1))
            zeta_traj[:, T:T+1] = zeta_term[:, mu:mu+1]
            Q_eff = Q + F.transpose() @ R @ F
            
            for k_math in range(T - 1, -1, -1):
                A_cl_curr_T = (A + B @ F + nu_traj[k_math] * A).transpose()
                zeta_traj[:, k_math:k_math+1] = A_cl_curr_T @ zeta_traj[:, k_math+1:k_math+2] + Q_eff @ y_traj[:, k_math+1:k_math+2]
                
            y0 = y_traj[:, 0:1]
            
            z0_sparse = F @ y0
            z0 = z0_sparse.toarray() if sparse.issparse(z0_sparse) else np.array(z0_sparse)
            
            s0_sparse = B.transpose() @ zeta_traj[:, 0:1]
            s0 = s0_sparse.toarray() if sparse.issparse(s0_sparse) else np.array(s0_sparse)
            
            E_sy_sum += s0 @ y0.transpose()
            E_zy_sum += z0 @ y0.transpose()
            zeta_term[:, mu:mu+1] = zeta_traj[:, 0:1]
            
        # 3. 梯度计算与下降
        E_sy = E_sy_sum / (T + 1)
        E_zy = E_zy_sum / (T + 1)
        
        R_full = R.toarray() if sparse.issparse(R) else np.array(R)
        mask_full = mask.toarray() if sparse.issparse(mask) else np.array(mask)
        
        # 施加网络拓扑掩码约束
        grad_J = (E_sy + R_full @ E_zy) * mask_full
        grad_norm = np.linalg.norm(grad_J, 'fro')
        
        # 4. 计算当前步的 Cost J
        F_current_full = F.toarray() if sparse.issparse(F) else np.array(F)
        F_curr_sparse = sparse.csr_matrix(F_current_full)
        
        Q_current = Q + F_curr_sparse.transpose() @ R @ F_curr_sparse
        A_cl_curr = A + B @ F_curr_sparse
        
        P = Q_current
        A_cl_curr_T = A_cl_curr.transpose()
        
        for i_lyap in range(5000):
            P_new = A_cl_curr_T @ P @ A_cl_curr + w * (A_T @ P @ A) + Q_current
            
            diff = P_new - P
            num_val = spla.norm(diff, 'fro') if sparse.issparse(diff) else np.linalg.norm(diff, 'fro')
            den_val = spla.norm(P, 'fro') if sparse.issparse(P) else np.linalg.norm(P, 'fro')
            
            if (num_val / (den_val + np.finfo(float).eps)) < 1e-5:
                break
            P = P_new
            
        cost_mat = P @ X_0
        J_record[ii - 1, 0] = cost_mat.diagonal().sum() if sparse.issparse(cost_mat) else np.trace(cost_mat)
        
        F = F - gam * sparse.csr_matrix(grad_J / grad_norm)
        
        if ii % 10 == 0:
            print(f"Dist-LargeScale Iter: {ii}, Grad: {grad_norm:.4e}, Cost: {J_record[ii - 1, 0]:.4e}")
            
        y_bar_init = y_bar[:, D + T : D + T + 1]
        
    F_dis_full = F.toarray() if sparse.issparse(F) else np.array(F)
    J_dis_full = J_record
    
    return F_dis_full, J_dis_full