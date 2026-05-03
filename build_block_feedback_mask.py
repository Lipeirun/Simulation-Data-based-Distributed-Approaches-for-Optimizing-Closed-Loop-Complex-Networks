import numpy as np

def build_block_feedback_mask(numAgents):
    # 边界约束校验
    if numAgents < 3:
        raise ValueError('子系统数量 numAgents 必须大于或等于 3。')

    # 1. 定义宏观拓扑邻接矩阵 (N x N)
    # 行 i 代表第 i 个子系统的控制输入 u_i
    # 列 j 代表第 j 个子系统的状态簇 x_j
    
    # 主对角线 (利用自身状态进行反馈控制)
    diag_struct = np.eye(numAgents)
    
    # 下副对角线 (利用前一个子系统的状态，实现前馈控制)
    subdiag_struct = np.diag(np.ones(numAgents - 1), -1)
    
    # 宏观拓扑矩阵合并
    base_topology = diag_struct + subdiag_struct
    
    # 注入循环边界条件：第 1 个子系统的控制器需要读取末端第 N 个子系统的状态
    base_topology[0, -1] = 1

    # 2. 维度映射膨胀
    # 目标是将 N x N 的宏观拓扑映射为 N x 3N 的 F 矩阵掩码
    # 即：如果 u_i 依赖 x_j 簇，则 F 矩阵对应行的 3 个元素均为 1
    block_F = np.array([1, 1, 1])
    
    # 使用张量积直接生成 N x 3N 的掩码结构
    mask_double = np.kron(base_topology, block_F)

    # 3. 强制类型转换：转为逻辑掩码 (布尔型) 以降低后续运算内存开销
    mask = mask_double.astype(bool)
    
    return mask