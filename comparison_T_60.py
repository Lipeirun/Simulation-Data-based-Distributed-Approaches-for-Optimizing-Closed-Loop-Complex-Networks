import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.io as sio
from types import SimpleNamespace

# 导入外部依赖模块
from chemistry_scalable import chemistry_scalable
from build_noise_model import build_noise_model
from build_block_feedback_mask import build_block_feedback_mask
from optimize_F_gradient_descent_full_large_scale import optimize_F_gradient_descent_full_large_scale
from optimize_F_gradient_descent_local_large_scale import optimize_F_gradient_descent_local_large_scale
from optimize_F_trajectory_full_large_scale import optimize_F_trajectory_full_large_scale
from optimize_F_trajectory_local_large_scale import optimize_F_trajectory_local_large_scale

def dlqr(A, B, Q, R):
    """等效实现 MATLAB 的 dlqr 函数"""
    P = la.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K

def main():
    np.random.seed(2026)

    # 1. 参数设置
    cfg = SimpleNamespace()
    cfg.numAgents  = 1000
    cfg.nxPerAgent = 3
    cfg.nuPerAgent = 1
    cfg.noiseState = 3
    cfg.gam = 0.07
    cfg.T = 60             # 仅设定单一预测时域 T=60
    cfg.D = 300
    cfg.dzm = 1000         # 迭代次数 (Update Time)
    cfg.w = 0.04           # 乘性噪声方差
    cfg.W = 0.1            # 加性噪声方差
    cfg.ex_w = -0.5        # 加性噪声均值

    # 2. 系统构造
    A, B = chemistry_scalable(cfg.numAgents)
    n = A.shape[0]
    m = B.shape[1]
    Q = np.eye(n)
    R = 1e-3 * np.eye(m)
    E_noise, X_0 = build_noise_model(cfg.numAgents, cfg.ex_w, cfg.W)
    mask = build_block_feedback_mask(cfg.numAgents)

    # 初始控制律
    F_lqr = -dlqr(A, B, Q, R) * 0.5
    F_sparse0 = F_lqr * mask

    # 3. 核心计算 (仅针对 T=60)
    # 3.1 获取梯度下降(Full Info & Local Info)的最优基准线
    _, J_opt_full_traj = optimize_F_gradient_descent_full_large_scale(A, B, F_lqr, Q, R, cfg.w, X_0, mask, cfg.gam, cfg.dzm)
    _, J_opt_local_traj = optimize_F_gradient_descent_local_large_scale(A, B, F_lqr, Q, R, cfg.w, X_0, mask, cfg.gam, cfg.dzm)

    # 提取收敛稳态值作为基准 (如果返回的是标量，直接赋值；如果是轨迹数组，取最后一个元素)
    opt_val_full = J_opt_full_traj if np.isscalar(J_opt_full_traj) else J_opt_full_traj[-1]
    opt_val_local = J_opt_local_traj if np.isscalar(J_opt_local_traj) else J_opt_local_traj[-1]

    # 3.2 轨迹采样优化 (T=60)
    F_final_full, J_dis_full = optimize_F_trajectory_full_large_scale(
        A, B, F_sparse0, mask, cfg.gam, cfg.dzm, cfg.T, cfg.D, Q, R, 
        cfg.w, cfg.numAgents, cfg.ex_w, cfg.W, X_0)
        
    F_final_local, J_dis_local = optimize_F_trajectory_local_large_scale(
        A, B, F_sparse0, mask, cfg.gam, cfg.dzm, cfg.T, cfg.D, Q, R, 
        cfg.w, cfg.numAgents, cfg.ex_w, cfg.W, X_0)

    # 4. 学术风绘图
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig = plt.figure(figsize=(6, 4.5), facecolor='w', dpi=100)
    ax = fig.add_subplot(111)
    
    x_axis = np.arange(1, cfg.dzm + 1)
    
    # 将返回的 J_dis 展平以匹配一维绘图需求
    J_dis_local_flat = J_dis_local.flatten()
    J_dis_full_flat = J_dis_full.flatten()

    # 严格按照截图颜色与线型重构
    p1, = ax.plot(x_axis, J_dis_local_flat, 'b-',  linewidth=2.0)      # Local info (蓝实线)
    p2, = ax.plot(x_axis, J_dis_full_flat,  'c--', linewidth=2.5)      # Full info (青虚线)
    p3, = ax.plot(x_axis, np.repeat(opt_val_local, cfg.dzm), 'r:', linewidth=2.0) # Optimal local (红点线)
    p4, = ax.plot(x_axis, np.repeat(opt_val_full, cfg.dzm),  'k:', linewidth=2.0) # Optimal full (黑点线)

    # 设置坐标轴与标签字体
    ax.set_xlabel('Update time', fontsize=13, fontname='Times New Roman')
    ax.set_ylabel('Cost Function', fontsize=13, fontname='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim([-50, cfg.dzm + 50])
    ax.grid(False)

    # 设置图例
    leg = ax.legend([p1, p2, p3, p4], 
                    ['Local info', 'Full info', 'Optimal cost (local info)', 'Optimal cost (full info)'],
                    loc='upper right', fontsize=11, frameon=True)

    plt.tight_layout()
    plt.show()

    # 5. 导出初始化环境与最优 F 矩阵
    save_filename = 'Simulation_Env_Data_T60.mat'
    sio.savemat(save_filename, {
        'A': A, 'B': B, 'Q': Q, 'R': R, 'mask': mask,
        'E_noise': E_noise, 'X_0': X_0, 
        'cfg': vars(cfg), 
        'F_sparse0': F_sparse0, 'F_lqr': F_lqr,
        'F_final_local': F_final_local, 'F_final_full': F_final_full
    })

    print(f'系统初始矩阵与分布式优化结果 F (T={cfg.T}) 已保存至: {save_filename}')

if __name__ == '__main__':
    main()