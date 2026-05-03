import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.linalg as la
from types import SimpleNamespace

# 导入同级目录下的自定义模块
from chemistry_scalable import chemistry_scalable
from build_noise_model import build_noise_model
from build_block_feedback_mask import build_block_feedback_mask
from optimize_F_trajectory_local_large_scale import optimize_F_trajectory_local_large_scale
from simulate_communication_time import simulate_communication_time

def dlqr(A, B, Q, R):
    """等效实现 MATLAB 的 dlqr 函数"""
    # 求解离散时间代数 Riccati 方程
    P = la.solve_discrete_are(A, B, Q, R)
    # 计算反馈增益矩阵 K
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K

def main():
    np.random.seed(2026)

    # 1. 实验参数与预分配
    agent_list = list(range(100, 2100, 100))
    num_tests = len(agent_list)

    time_central_list = np.zeros(num_tests)
    time_dist_list = np.zeros(num_tests)

    # 固定超参数
    cfg = SimpleNamespace()
    cfg.nxPerAgent = 3
    cfg.nuPerAgent = 1
    cfg.noiseState = 3
    cfg.gam = 0.07
    cfg.T = 60
    cfg.D = 300
    cfg.dzm = 1            # 梯度下降的次数
    cfg.w = 0.04           # 乘性噪声的方差
    cfg.W = 0.1            # 加性噪声的方差
    cfg.ex_w = -0.5        # 加性噪声的均值

    # 2. 循环遍历不同规模的子系统
    for idx in range(num_tests):
        cfg.numAgents = agent_list[idx]
        print(f'\n---> 测试系统规模: numAgents = {cfg.numAgents}')
        
        # --- 系统重构 ---
        A, B = chemistry_scalable(cfg.numAgents)
        n = A.shape[0]
        m = B.shape[1]
        Q = np.eye(n)
        R = 1e-3 * np.eye(m)
        E_noise, X_0 = build_noise_model(cfg.numAgents, cfg.ex_w, cfg.W)
        mask = build_block_feedback_mask(cfg.numAgents)
        
        print('     正在求解初始 LQR...')
        F_lqr = -dlqr(A, B, Q, R) * 0.5
        F_sparse0 = F_lqr * mask  # 施加掩码
        
        # --- 2) 分布式局部：compute serial ---
        print('     正在运行分布式优化 ...')
        start_time = time.time()
        _, _ = optimize_F_trajectory_local_large_scale(
            A, B, F_sparse0, mask, cfg.gam, cfg.dzm, cfg.T, cfg.D, 
            Q, R, cfg.w, cfg.numAgents, cfg.ex_w, cfg.W, X_0
        )
        time_local_compute_serial = time.time() - start_time
        
        # --- 3) 通信时间模拟与分布式真实耗时推算 ---
        num_comm_rounds_per_dzm = (cfg.D + cfg.T) + (cfg.T + 1) * (2 * cfg.T)
        total_comm_rounds = cfg.dzm * num_comm_rounds_per_dzm
        m_comm = cfg.nxPerAgent 
        
        t_comm_per_round_sec = simulate_communication_time(m_comm, 1)
        time_communication_total = total_comm_rounds * t_comm_per_round_sec
        
        time_local_compute_parallel = time_local_compute_serial / cfg.numAgents
        time_dist_list[idx] = time_local_compute_parallel + time_communication_total
        time_central_list[idx] = time_local_compute_parallel  # 临时 
        
        print(f'     当前规模测试完成。Central: {time_central_list[idx]:.4f} s | Dist: {time_dist_list[idx]:.4f} s')

    # 3. 绘制单独绘制分布式耗时折线图（对应原图3）
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    fig = plt.figure(figsize=(6, 4.5), facecolor='w', dpi=100)
    ax = fig.add_subplot(111)
    
    ax.plot(agent_list, time_dist_list, 'b-s', linewidth=2.0, markersize=6, markerfacecolor='b')
    
    ax.set_xlabel('Number of Subsystems', fontsize=13, fontname='Times New Roman')
    ax.set_ylabel('Computation Time (s)', fontsize=13, fontname='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()