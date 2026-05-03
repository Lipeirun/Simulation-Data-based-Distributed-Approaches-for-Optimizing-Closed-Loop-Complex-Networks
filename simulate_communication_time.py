def simulate_communication_time(m: int, num_exchanges_per_step: int = 1) -> float:
    """
    计算基于实际硬件延迟参数的点对点网络通信时间。
    """
    # 论文在 1-Gbps Ethernet 测量的底层点对点硬件参数
    alpha = 0.436         # 握手与路由延迟 (ms)
    beta = 3.6e-5         # 单个元素传输物理时间 (ms)
    
    # 在理想并发的局部拓扑下，单步网络通信时间只取决于单条链路的点对点传输时间
    t_comm_ms_per_exchange = alpha + m * beta
    
    t_comm_ms_total = t_comm_ms_per_exchange * num_exchanges_per_step
    
    # 转为秒
    t_comm_seconds = t_comm_ms_total / 1000.0
    
    return t_comm_seconds