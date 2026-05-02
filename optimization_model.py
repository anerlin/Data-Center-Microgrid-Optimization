"""
论文 Section III：日前优化调度模型
"""
import pulp
import numpy as np
from workload_shift import WorkloadShiftModel

class DCMOptimizationModel:
   def __init__(self, T, params, loads, wind_forecast, pv_forecast, temp_out):
    self.T = T
    self.params = params
    self.loads = loads
    # 截断数据到 T 个点
    self.wind = wind_forecast[:T]
    self.pv = pv_forecast[:T]
    self.temp_out = temp_out[:T]
    
    # 同时截断 loads 里的所有曲线
    for key in self.loads:
        self.loads[key] = self.loads[key][:T]
    
    N_servers = params['N_racks'] * params['N_servers_per_rack']
    self.D_dc_max = N_servers * params['N_CPU_per_server'] * params['D_cpu_max']
        
    def build_model(self):
        """构建完整优化模型"""
        model = pulp.LpProblem("DCM_Optimization", pulp.LpMinimize)
        
        # ========== 决策变量 ==========
        P_buy = [pulp.LpVariable(f"P_buy_{t}", 0, 360) for t in range(self.T)]
        P_sell = [pulp.LpVariable(f"P_sell_{t}", 0, 100) for t in range(self.T)]
        buy_flag = [pulp.LpVariable(f"buy_flag_{t}", cat='Binary') for t in range(self.T)]
        sell_flag = [pulp.LpVariable(f"sell_flag_{t}", cat='Binary') for t in range(self.T)]
        
        P_wind = [pulp.LpVariable(f"P_wind_{t}", 0, self.wind[t]) for t in range(self.T)]
        P_pv = [pulp.LpVariable(f"P_pv_{t}", 0, self.pv[t]) for t in range(self.T)]
        
        P_data = [pulp.LpVariable(f"P_data_{t}", 0, self.params['P_peak']) for t in range(self.T)]
        P_refri = [pulp.LpVariable(f"P_refri_{t}", 0, self.params['P_refri_max']) for t in range(self.T)]
        
        # ========== 负载平移 ==========
        shift_model = WorkloadShiftModel(self.T, self.params)
        
        model, D_A_shift, C_data = shift_model.add_short_running_shifts(
            model, self.loads['A'])
        
        model, D_B1_shift = shift_model.add_long_continuous_shifts(
            model, self.loads['B1'], self.params['t_e_B1'], self.params['t_l_B1'])
        model, D_B2_shift = shift_model.add_long_continuous_shifts(
            model, self.loads['B2'], self.params['t_e_B2'], self.params['t_l_B2'])
        
        model, D_C1_shift = shift_model.add_long_interruptible_shifts(
            model, self.loads['C1'], self.params['C_range'])
        model, D_C2_shift = shift_model.add_long_interruptible_shifts(
            model, self.loads['C2'], self.params['C_range'])
        model, D_C3_shift = shift_model.add_long_interruptible_shifts(
            model, self.loads['C3'], self.params['C_range'])
        
        # ========== 约束条件 ==========
        M = 1e6
        
        for t in range(self.T):
            total_load = (D_A_shift[t] + D_B1_shift[t] + D_B2_shift[t] +
                         D_C1_shift[t] + D_C2_shift[t] + D_C3_shift[t] +
                         self.loads['MG'][t])
            
            # 修复：用 min 约束代替 if 判断
            u_cpu = total_load / self.D_dc_max
            u_cpu_clip = pulp.LpVariable(f"u_cpu_clip_{t}", 0, 1)
            model += u_cpu_clip <= u_cpu
            model += u_cpu_clip <= 1
            
            model += P_data[t] == self.params['P_idle'] + (
                self.params['P_peak'] - self.params['P_idle']) * u_cpu_clip
            
            Q_refri = P_data[t] + self.params['beta'] * self.params['Sa']
            model += P_refri[t] >= Q_refri / self.params['eta_refri']
            
            model += P_wind[t] + P_pv[t] + P_buy[t] == P_data[t] + P_refri[t] + P_sell[t]
            
            model += P_buy[t] <= M * buy_flag[t]
            model += P_sell[t] <= M * sell_flag[t]
            model += buy_flag[t] + sell_flag[t] <= 1
        
        # ========== 目标函数 ==========
        cost_buy = pulp.lpSum(self.params['K_buy'] * P_buy[t] * self.params['dt'] 
                              for t in range(self.T))
        revenue_sell = -pulp.lpSum(self.params['K_sell'] * P_sell[t] * self.params['dt']
                                    for t in range(self.T))
        curtail_cost = pulp.lpSum(self.params['K_cur'] * (
            self.wind[t] - P_wind[t] + self.pv[t] - P_pv[t]) * self.params['dt']
            for t in range(self.T))
        
        model += cost_buy + revenue_sell + curtail_cost + C_data
        
        return model
    
    def solve(self):
        """求解模型"""
        model = self.build_model()
        
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300)
        model.solve(solver)
        
        if model.status == pulp.LpStatusOptimal:
            return model
        else:
            print(f"无可行解，状态: {model.status}")
            return None
