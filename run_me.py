"""
数据中心微网优化调度 - 真实7条曲线 + 论文参数
"""
import numpy as np
import pandas as pd
import pulp
import os

print("="*60)
print("数据中心微网优化调度复现（完整7条曲线）")
print("="*60)

T = 96
dt = 0.25

P_peak = 360
P_idle = 216
D_dc_max = 3.84e6

# 论文原参数
K_buy = 0.8
K_sell = 0.4
K_cur = 2.0

csv_path = "data/figure2_workloads.csv"

if os.path.exists(csv_path):
    print(f"\n读取CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV列名: {list(df.columns)}")
    
    # 7条曲线
    required_curves = [
        'Interactive workloads',
        'Short-running workloads A',
        'Long-running continuous workload B1',
        'Long-running continuous workload B2',
        'Long-running interruptible workload C1',
        'Long-running interruptible workload C2',
        'Long-running interruptible workload C3'
    ]
    
    short_names = ['MG', 'A', 'B1', 'B2', 'C1', 'C2', 'C3']
    
    loads = {}
    for full_name, short_name in zip(required_curves, short_names):
        if full_name in df.columns:
            loads[short_name] = df[full_name].values[:T]
            print(f"  ✅ {full_name}")
        else:
            print(f"  ❌ 缺失: {full_name}")
            loads[short_name] = np.zeros(T)
    
    total_load = loads['MG'] + loads['A'] + loads['B1'] + loads['B2'] + \
                 loads['C1'] + loads['C2'] + loads['C3']
else:
    print(f"CSV不存在: {csv_path}")
    exit()

# 新能源数据
t = np.arange(96)
wind = 500 * (0.4 + 0.3 * np.sin(np.pi * (t - 24) / 48))
wind = np.maximum(0, wind)[:T]
pv = 500 * np.maximum(0, np.sin(np.pi * t / 48))[:T]

print(f"\n时段数: {T}, 每时段: {dt}小时")

# 简单调度
print("\n1. 简单调度（只从电网买电）...")
simple_cost = 0
for t in range(T):
    load_ratio = total_load[t] / D_dc_max
    load_ratio = min(load_ratio, 1.0)
    P_server = P_idle + (P_peak - P_idle) * load_ratio
    simple_cost += P_server * K_buy * dt
print(f"   成本: ¥{simple_cost:.2f}")

# 优化调度
print("\n2. 优化调度（新能源 + 购售电）...")

model = pulp.LpProblem("DCM", pulp.LpMinimize)

P_buy = [pulp.LpVariable(f"buy_{t}", 0, 360) for t in range(T)]
P_sell = [pulp.LpVariable(f"sell_{t}", 0, 100) for t in range(T)]
P_wind = [pulp.LpVariable(f"wind_{t}", 0, wind[t]) for t in range(T)]
P_pv = [pulp.LpVariable(f"pv_{t}", 0, pv[t]) for t in range(T)]
P_server = [pulp.LpVariable(f"server_{t}", 0, P_peak) for t in range(T)]

for t in range(T):
    load_ratio = total_load[t] / D_dc_max
    load_ratio = min(load_ratio, 1.0)
    model += P_server[t] == P_idle + (P_peak - P_idle) * load_ratio
    model += P_wind[t] + P_pv[t] + P_buy[t] == P_server[t] + P_sell[t]

obj = (sum(K_buy * P_buy[t] * dt for t in range(T)) -
       sum(K_sell * P_sell[t] * dt for t in range(T)) +
       sum(K_cur * ((wind[t] - P_wind[t]) + (pv[t] - P_pv[t])) * dt for t in range(T)))

model += obj
model.solve(pulp.PULP_CBC_CMD(msg=False))

if model.status == pulp.LpStatusOptimal:
    opt_cost = pulp.value(model.objective)
    print(f"   成本: ¥{opt_cost:.2f}")
    reduction = simple_cost - opt_cost
    print(f"\n成本降低: ¥{reduction:.2f} ({reduction/simple_cost*100:.1f}%)")
else:
    print(f"   无可行解")

print("\n完成！")