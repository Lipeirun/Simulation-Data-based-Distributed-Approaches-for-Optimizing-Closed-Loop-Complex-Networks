import numpy as np

def chemistry_scalable(numAgents=3):
    # 默认配置与边界约束校验
    if numAgents < 3:
        raise ValueError('系统数量 numAgents 至少为 3 才能维持 头-中-尾 结构')

    # --- 1. 参数定义 ---
    d_T = 0.1
    
    # 提取并分配稳态工作点变量 (根据原代码扩展)
    # 格式 [xA, xB, T]
    x_sys1 = np.array([0.6, 0.352, 327])
    x_sysMid = np.array([0.536, 0.4, 328.4])
    x_sysN = np.array([0.285, 0.565, 328.5])
    
    x = np.zeros(3 * numAgents)
    x[0:3] = x_sys1
    for i in range(1, numAgents - 1):
        x[3*i : 3*i+3] = x_sysMid
    x[3*numAgents-3 : 3*numAgents] = x_sysN

    # 物理与动力学参数
    F_10 = 8.3
    F_mid0 = 0.5
    F_r = 40
    F_p = 0.01 * F_r
    V1 = 89.4
    V_mid = 90
    VN = 13.27
    k1 = 0.336
    k2 = 0.089
    E1_R = -100
    E2_R = -150
    Delta_H1 = -40
    Delta_H2 = -50
    C_p = 2.5
    rou = 0.15
    alpha_A = 3.5
    alpha_B = 1.1
    alpha_C = 0.5

    # 流量累积计算：F_sys[i] 表示流入第 i+1 级的总内部流量 (Python 从 0 索引)
    F_sys = np.zeros(numAgents)
    F_sys[0] = F_10 + F_r
    for i in range(1, numAgents - 1):
        F_sys[i] = F_sys[i-1] + F_mid0

    # 计算末端反馈循环的相关偏导参数
    xA_N = x[3*numAgents-3]
    xB_N = x[3*numAgents-2]
    m = (alpha_A * xA_N + alpha_B * xB_N + alpha_C * (1 - xA_N - xB_N))
    m_AA = (alpha_A * m - alpha_A**2 * xA_N) / m**2
    m_AB = -alpha_A * xA_N * alpha_B / m**2
    m_BA = -alpha_B * xB_N * alpha_A / m**2
    m_BB = (alpha_B * m - alpha_B**2 * xB_N) / m**2

    # --- 2. 连续时间矩阵构建 ---
    A_cont = np.zeros((3 * numAgents, 3 * numAgents))
    B_cont = np.zeros((3 * numAgents, numAgents))

    for i in range(numAgents):
        # 当前级状态变量索引 (映射到 Python 的 0-based 索引)
        idx_xA = 3 * i
        idx_xB = 3 * i + 1
        idx_T = 3 * i + 2
        
        xA_i = x[idx_xA]
        xB_i = x[idx_xB]
        T_i = x[idx_T]
        
        # 动力学常数
        k1_exp = k1 * np.exp(-E1_R / T_i)
        k2_exp = k2 * np.exp(-E2_R / T_i)

        if i == 0:
            # --- Sys 1 (头节点) ---
            # 内部动力学
            A_cont[0, 0] = -F_10/V1 - F_r/V1 - k1_exp
            A_cont[0, 2] = -k1_exp * xA_i * E1_R / T_i**2
            A_cont[1, 0] = k1_exp
            A_cont[1, 1] = -F_10/V1 - F_r/V1 - k2_exp
            A_cont[1, 2] = xA_i * k1_exp * E1_R / T_i**2
            A_cont[2, 0] = -Delta_H1/C_p * k1_exp
            A_cont[2, 1] = -Delta_H2/C_p * k2_exp
            A_cont[2, 2] = -F_10/V1 - F_r/V1 - Delta_H1/C_p * k1_exp * E1_R / T_i**2 * xA_i \
                           - Delta_H2/C_p * k2_exp * E2_R / T_i**2 * xB_i
            
            # 尾部反馈 (来自 Sys N)
            idx_xAN = 3 * numAgents - 3
            idx_xBN = 3 * numAgents - 2
            idx_TN  = 3 * numAgents - 1
            A_cont[0, idx_xAN] = F_r/V1 * m_AA
            A_cont[0, idx_xBN] = F_r/V1 * m_AB
            A_cont[1, idx_xAN] = F_r/V1 * m_BA
            A_cont[1, idx_xBN] = F_r/V1 * m_