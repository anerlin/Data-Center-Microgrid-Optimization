import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from optimization_model import DCMOptimizationModel

def run_simple_scheduling(loader):
    """简单调度：不移负载"""
    params = loader.load_parameters()
    loads = loader.load_workload_curves()
    
    T = 48
    dt = 0.25
    total_load = loads['MG'] + loads['A'] + loads['B1'] + loads['B2'] + \
                 loads['C1'] + loads['C2'] + loads['C3']
    
    cost = 0
    for t in range(T):
        load_ratio = total_load[t] / params['D_dc_max']
        load_ratio = min(load_ratio, 1.0)
        P_needed = params['P_idle'] + (params['P_peak'] - params['P_idle']) * load_ratio
        cost += P_needed * params['K_buy'] * dt
    
    print(f"简单调度（不移负载）成本: ¥{cost:.2f}")
    return cost

def run_optimization_scheduling(loader):
    """优化调度：考虑负载平移"""
    params = loader.load_parameters()
    loads = loader.load_workload_curves()
    wind, pv = loader.load_renewable_forecast()
    temp = loader.load_temperature()
    
    model = DCMOptimizationModel(96, params, loads, wind, pv, temp)
    result = model.solve()
    
    if result:
        print(f"\n优化调度（移负载）成本: ¥{result.objective_value:.2f}")
        return result.objective_value
    else:
        return None

def main():
    print("=" * 60)
    print("数据中心微网优化调度复现")
    print("=" * 60)
    
    loader = DataLoader()
    
    print("\n1. 运行简单调度（不移动负载）...")
    simple_cost = run_simple_scheduling(loader)
    
    print("\n2. 运行优化调度（移动负载）...")
    opt_cost = run_optimization_scheduling(loader)
    
    if opt_cost:
        reduction = (simple_cost - opt_cost) / simple_cost * 100
        print(f"\n成本降低: ¥{simple_cost - opt_cost:.2f} ({reduction:.1f}%)")
    else:
        print("\n优化调度无可行解，请检查约束条件")
    
    print("\n完成！")

if __name__ == "__main__":
    main()