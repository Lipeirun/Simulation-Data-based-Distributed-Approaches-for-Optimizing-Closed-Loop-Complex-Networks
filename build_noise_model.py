import numpy as np

def build_noise_model(numAgents, ex_w, W):
    if W < 0:
        raise ValueError('方差 W 必须大于或等于 0。')

    n = 3 * numAgents
    E_noise = np.zeros((n, 1))
    
    # 生成实时噪声向量
    sigma = np.sqrt(W)
    noise_T = np.random.normal(ex_w, sigma, (numAgents, 1))
    
    # 将噪声分配到所有的 T 状态上 (映射 MATLAB 的 3:3:end，即索引 2, 5, 8...)
    E_noise[2::3, 0:1] = noise_T
    
    # 构造期望向量 mu = E[x_0]
    mu = np.zeros((n, 1))
    mu[2::3, 0] = ex_w
    
    # 构造方差对角矩阵 Sigma
    Sigma = np.zeros((n, n))
    for i in range(numAgents):
        # 对应 MATLAB 的 3*i 索引 (转换为 Python 为 3*i+2)
        Sigma[3*i+2, 3*i+2] = W
    
    # X_0 = mu * mu^T + Sigma
    X_0 = mu @ mu.T + Sigma
    
    return E_noise, X_0