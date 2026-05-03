import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from types import SimpleNamespace

# 导入外部依赖模块
from chemistry_scalable import chemistry_scalable
from build_noise_model import build_noise_model
from build_block_feedback_mask import build_block_feedback_mask
from optimize_F_gradient_descent_local_large_scale import optimize_F_gradient_descent_local_large_scale
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
    cfg.gam = 0.1
    cfg.T = [3, 20, 60]
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
    
    # 记录总的结果
    _, J_opt_local_traj = optimize_F_gradient_descent_local_large_scale(A, B, F_lqr, Q, R, cfg.w, X_0, mask, cfg.gam, cfg.dzm)
    J_opt_local = J_opt_local_traj if np.isscalar(J_opt_local_traj) else J_opt_local_traj[-1]

    J_dis_local = np.zeros((cfg.dzm, 3))
    
    for ii in range(3):
        T_val = cfg.T[ii]
        _, J_dis_local_temp = optimize_F_trajectory_local_large_scale(
            A, B, F_sparse0, mask, cfg.gam, cfg.dzm, T_val, cfg.D, Q, R, 
            cfg.w, cfg.numAgents, cfg.ex_w, cfg.W, X_0)
        J_dis_local[:, ii] = J_dis_local_temp.flatten()

    # 3. 学术风绘图
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig = plt.figure(figsize=(6, 4.5), facecolor='w', dpi=100)
    ax = fig.add_subplot(111)
    
    x_axis = np.arange(1, cfg.dzm + 1)

    # 绘制 T=3, T=20, T=60 的轨迹
    p1, = ax.plot(x_axis, J_dis_local[:, 0], 'b-',  linewidth=2.0)
    p2, = ax.plot(x_axis, J_dis_local[:, 1], 'c--', linewidth=2.5)
    p3, = ax.plot(x_axis, J_dis_local[:, 2], 'm-.', linewidth=2.5)

    # 处理最优代价基准线
    opt_val = J_opt_local 
    p4, = ax.plot(x_axis, np.repeat(opt_val, cfg.dzm), 'k:', linewidth=1.5)

    # 设置坐标轴与标签字体
    ax.set_xlabel('Update time', fontsize=13, fontname='Times New Roman')
    ax.set_ylabel('Cost Function', fontsize=13, fontname='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim([-50, cfg.dzm + 50])
    ax.grid(False)

    # 设置图例
    leg = ax.legend([p1, p2, p3, p4], ['T=3', 'T=20', 'T=60', 'Optimal cost (local info)'],
                    loc='upper right', fontsize=11, frameon=True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()