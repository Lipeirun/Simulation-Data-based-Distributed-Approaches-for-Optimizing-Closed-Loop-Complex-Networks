import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def main():
    plt.close('all')
    np.random.seed(2026)

    # 设定 matplotlib 字体参数以匹配 Times New Roman 与 LaTeX 风格
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm' # 使用类似 LaTeX 的数学字体

    # 加载数据 (simplify_cells 自动解除 MATLAB 繁复的结构体嵌套)
    mat_data = sio.loadmat('Simulation_Env_Data_T60.mat', simplify_cells=True)
    
    cfg = mat_data['cfg']
    total_agents = int(cfg['numAgents'])
    nx = int(cfg['nxPerAgent'])
    cfg_w = float(cfg['w'])
    
    A = mat_data['A']
    B = mat_data['B']
    F_final_local = mat_data['F_final_local']
    X_0 = mat_data['X_0']
    
    num_plot = 20
    dt = 0.1                   
    T_sim = 10                 
    N_steps = round(T_sim / dt) + 1
    time_axis = np.arange(N_steps) * dt

    n = A.shape[0]
    m = B.shape[1]

    X_traj = np.zeros((n, N_steps))
    U_traj = np.zeros((m, N_steps))

    # [V, D] = eig(full(X_0)) 
    D, V = np.linalg.eig(X_0)
    D = np.real(D)
    D[D < 0] = 0
    V = np.real(V)
    
    # x_init = V * sqrt(D) * randn(n, 1) (注意 Python eig 的 D 返回是 1D 数组，需转为对角阵)
    x_init = V @ np.diag(np.sqrt(D)) @ np.random.randn(n, 1)

    X_traj[:, 0:1] = x_init
    U_traj[:, 0:1] = F_final_local @ x_init

    for k in range(N_steps - 1):
        nu_k = np.random.normal(0, np.sqrt(cfg_w))
        X_traj[:, k+1:k+2] = (A + B @ F_final_local + nu_k * A) @ X_traj[:, k:k+1]
        U_traj[:, k+1:k+2] = F_final_local @ X_traj[:, k+1:k+2]

    # agent_ids = sort(randperm(total_agents, num_plot));
    # 注意：Python 从 0 索引，因此产生的 id 是 0 到 total_agents-1
    agent_ids = np.sort(np.random.choice(total_agents, num_plot, replace=False))

    T_dev = np.zeros((num_plot, N_steps))
    Q_dev = np.zeros((num_plot, N_steps))

    for i in range(num_plot):
        aid = agent_ids[i]
        # MATLAB: X_traj(nx * agent_ids(i), :) -> 对于 0-based 索引，提取对应 agent 的最后一个状态
        T_dev[i, :] = X_traj[nx * aid + nx - 1, :]
        Q_dev[i, :] = U_traj[aid, :]

    colors = plt.cm.get_cmap('tab20').colors # 替代 MATLAB 的 lines 调色板
    leg_X_str = []
    leg_U_str = []
    num_leg_cols = max(1, int(np.ceil(num_plot / 5)))

    # --- 图 1：状态 X 轨迹 ---
    fig1 = plt.figure(figsize=(6, 4.5), facecolor='w', dpi=100)
    ax1 = fig1.add_subplot(111)
    
    for i in range(num_plot):
        ax1.plot(time_axis, T_dev[i, :], color=colors[i % len(colors)], linewidth=2.0)
        # 标签使用 aid + 1 以对齐 MATLAB 从 1 开始的直观编号
        leg_X_str.append(rf'$x_{{{agent_ids[i] + 1},3}}$')

    ax1.set_xlabel('Time (s)', fontsize=13, fontname='Times New Roman')
    ax1.set_ylabel(r'$x_{i,3}$', fontsize=14)
    ax1.set_xlim([0, T_sim])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    leg1 = ax1.legend(leg_X_str, loc='best', ncol=num_leg_cols, frameon=True, fontsize=12)
    
    # --- 图 2：控制 U 轨迹 ---
    fig2 = plt.figure(figsize=(6, 4.5), facecolor='w', dpi=100)
    ax2 = fig2.add_subplot(111)
    
    for i in range(num_plot):
        ax2.plot(time_axis, Q_dev[i, :], color=colors[i % len(colors)], linewidth=2.0)
        leg_U_str.append(rf'$u_{{{agent_ids[i] + 1}}}$')

    ax2.set_xlabel('Time (s)', fontsize=13, fontname='Times New Roman')
    ax2.set_ylabel(r'$u_i$', fontsize=14)
    ax2.set_xlim([0, T_sim])
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    leg2 = ax2.legend(leg_U_str, loc='best', ncol=num_leg_cols, frameon=True, fontsize=12)

    plt.show()

if __name__ == '__main__':
    main()