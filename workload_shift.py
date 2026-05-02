"""
论文 Section II：三类延迟容忍负载的时间平移建模
"""
import pulp
import numpy as np

class WorkloadShiftModel:
    def __init__(self, T, params):
        self.T = T  # 96时段
        self.params = params
        
    def add_short_running_shifts(self, model, D_A_orig):
        """
        短运行可延迟负载 A（公式3-9）
        特点：可以部分/整体转移到后续 H 个时段
        """
        H = 6
        D_A_max = np.max(D_A_orig)
        
        # 转移矩阵 U_A[t][k]（公式3）
        U_A = {}
        for t in range(self.T):
            for k in range(self.T):
                # 只能向后转移，且在 H 时段内
                if 0 <= (t - k) <= H:
                    U_A[(t, k)] = pulp.LpVariable(f"U_A_{t}_{k}", 0, D_A_max)
                # 跨天循环转移（允许）
                elif t < k and (self.T - k + t) <= H:
                    U_A[(t, k)] = pulp.LpVariable(f"U_A_{t}_{k}", 0, D_A_max)
        
        # 公式4：原始负载 = 转出的总量
        for k in range(self.T):
            outflow = pulp.lpSum(U_A[(t, k)] for t in range(self.T) if (t, k) in U_A)
            model += outflow == D_A_orig[k]
        
        # 公式5：转移后负载 = 转入的总量
        D_A_shift = []
        for t in range(self.T):
            inflow = pulp.lpSum(U_A[(t, k)] for k in range(self.T) if (t, k) in U_A)
            D_A_shift.append(inflow)
        
        # 公式8：负载平移惩罚（避免无效转移）
        C_data = 0
        for t in range(self.T):
            for k in range(self.T):
                if (t, k) in U_A:
                    # 转移越远，惩罚越大
                    delay = min(abs(t - k), self.T - abs(t - k))
                    C_data += self.params['K_delay'] * U_A[(t, k)] * 0.25 * delay
        
        return model, D_A_shift, C_data
    
    def add_long_continuous_shifts(self, model, D_B_orig, t_e, t_l):
        """
        长运行连续负载 B（公式10-16）
        特点：整体平移，形状不变
        """
        K = len(D_B_orig)  # 负载的持续时段数
        
        # 开始时间变量 U_B（公式11）
        U_B = [pulp.LpVariable(f"U_B_{t}", cat='Binary') for t in range(self.T)]
        model += pulp.lpSum(U_B) == 1  # 公式12
        
        # 公式13-14：最早/最晚开始时间约束
        start_time = pulp.lpSum([t * U_B[t] for t in range(self.T)])
        model += start_time >= t_e
        model += start_time <= t_l
        
        # 公式15-16：构造平移后的负载
        D_B_shift = []
        for t in range(self.T):
            expr = 0
            for start in range(self.T):
                idx = t - start
                if 0 <= idx < K:
                    expr += U_B[start] * D_B_orig[idx]
            D_B_shift.append(expr)
        
        return model, D_B_shift
    
    def add_long_interruptible_shifts(self, model, D_C_orig, time_range):
        """
        长运行可中断负载 C（公式17-23）
        特点：拆成多个子任务，顺序不变
        """
        K = len(D_C_orig)  # 子任务数
        
        # 转移矩阵 U_C[k][t]（公式18）
        U_C = {}
        for k in range(K):
            for t in range(time_range[0], time_range[1]+1):
                if t < self.T:
                    U_C[(k, t)] = pulp.LpVariable(f"U_C_{k}_{t}", cat='Binary')
        
        # 公式19：每个子任务只能去一个时间点
        for k in range(K):
            model += pulp.lpSum(U_C[(k, t)] for t in range(self.T) if (k, t) in U_C) == 1
        
        # 公式22：子任务顺序约束（k+1 > k）
        for k in range(K-1):
            t_k = pulp.lpSum(t * U_C[(k, t)] for t in range(self.T) if (k, t) in U_C)
            t_k1 = pulp.lpSum(t * U_C[(k+1, t)] for t in range(self.T) if (k+1, t) in U_C)
            model += t_k1 >= t_k + 1
        
        # 公式23：构造转移后的负载
        D_C_shift = []
        for t in range(self.T):
            expr = pulp.lpSum(U_C[(k, t)] * D_C_orig[k] for k in range(K) if (k, t) in U_C)
            D_C_shift.append(expr)
        
        return model, D_C_shift