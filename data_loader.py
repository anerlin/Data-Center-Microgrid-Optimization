"""
数据加载模块
优先读取 Engauge Digitizer 导出的 CSV 真实数据
如果 CSV 不存在，则使用模拟数据
"""
import numpy as np
import os

class DataLoader:
    def __init__(self, data_folder=None):
        self.T = 48  # 24小时 × 2 = 48个时段（每15分钟）
        self.dt = 0.25  # 小时
        
        # 设置数据文件夹
        if data_folder is None:
            # 默认：项目根目录下的 data 文件夹
            self.data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        else:
            self.data_folder = data_folder
        
        # 确保 data 文件夹存在
        os.makedirs(self.data_folder, exist_ok=True)
        
    def load_workload_curves(self):
        """
        加载图2的7类负载曲线
        优先从 CSV 读取，如果没有则使用模拟数据
        """
        csv_path = os.path.join(self.data_folder, 'figure2_workloads.csv')
        
        if os.path.exists(csv_path):
            print(f"  从 CSV 读取真实数据: {csv_path}")
            return self._load_from_csv(csv_path)
        else:
            print(f"  CSV 文件不存在，使用模拟数据")
            print(f"  请将 Engauge Digitizer 导出的文件保存为: {csv_path}")
            return self._generate_mock_data()
    
    def _load_from_csv(self, csv_path):
        """从 Engauge Digitizer 导出的 CSV 读取数据"""
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        # 获取列名（去除空列）
        columns = [col for col in df.columns if 'Unnamed' not in col and col != 'Point Index']
        
        # 列名映射（根据你的 CSV 实际列名调整）
        name_mapping = {
            'Interactive workloads': 'MG',
            'Short-running workloads A': 'A',
            'Long-running continuous workload B1': 'B1',
            'Long-running continuous workload B2': 'B2',
            'Long-running interruptible workload C1': 'C1',
            'Long-running interruptible workload C2': 'C2',
            'Long-running interruptible workload C3': 'C3',
        }
        
        loads = {}
        for col in columns:
            for eng_name, short_name in name_mapping.items():
                if eng_name in col or short_name in col:
                    loads[short_name] = df[col].values[:self.T]
                    break
        
        # 检查是否所有负载都读取到了
        expected = ['MG', 'A', 'B1', 'B2', 'C1', 'C2', 'C3']
        for key in expected:
            if key not in loads:
                print(f"  警告: 未找到曲线 {key}，使用模拟数据")
                mock = self._generate_mock_data()
                return mock
        
        return loads
    
    def _generate_mock_data(self):
        """生成模拟数据（接近论文图2）"""
        t = np.arange(self.T)
        
        # Interactive workloads (MG) - 白昼型
        D_MG = 0.4e6 + 0.7e6 * np.sin(np.pi * (t - 24) / 48)
        D_MG = np.maximum(0.2e6, D_MG)
        
        # Short-running A - 平稳型
        D_A = 0.22e6 * np.ones(self.T)
        D_A[60:80] = 0.1e6
        
        # B1 - 集中在40-56时段
        D_B1 = np.zeros(self.T)
        D_B1[40:56] = 0.14e6
        
        # B2 - 集中在44-62时段
        D_B2 = np.zeros(self.T)
        D_B2[44:62] = 0.09e6
        
        # C1 - 20-35时段
        D_C1 = np.zeros(self.T)
        D_C1[20:35] = 0.07e6
        
        # C2 - 30-45时段
        D_C2 = np.zeros(self.T)
        D_C2[30:45] = 0.05e6
        
        # C3 - 60-75时段
        D_C3 = np.zeros(self.T)
        D_C3[60:75] = 0.09e6
        # 截取前 self.T 个点
        return {
            'MG': D_MG[:self.T],
            'A': D_A[:self.T],
            'B1': D_B1[:self.T],
            'B2': D_B2[:self.T],
            'C1': D_C1[:self.T],
            'C2': D_C2[:self.T],
            'C3': D_C3[:self.T],
        }    
    def load_renewable_forecast(self):
        """加载图3的风电/光伏预测曲线"""
        t = np.arange(self.T)
        
        # 风电：夜间高，白天低
        wind = 500 * (0.4 + 0.3 * np.sin(np.pi * (t - 24) / 48))
        wind = np.maximum(0, wind)
        
        # 光伏：白天高，夜间0
        pv = 500 * np.maximum(0, np.sin(np.pi * t / 48))
        
        return wind, pv
    
    def load_temperature(self):
        """加载附录图17的室外温度曲线"""
        t = np.arange(self.T)
        temp = 19 + 9 * np.sin(np.pi * (t - 24) / 48)
        temp = np.maximum(10, temp)
        return temp
    
    def load_parameters(self):
        """加载表1、表2的参数"""
        params = {
            # 数据中心参数（表1）
	    'dt': 0.25,
            'N_racks': 60,
            'N_servers_per_rack': 10,
            'N_CPU_per_server': 4,
            'D_cpu_max': 1600,
            'P_cpu_max': 0.15,
            'P_peak': 360,
            'P_idle': 216,
            
            # 制冷参数
            'eta_refri': 3.5,
            'P_refri_max': 143,
            'beta': 0.7,
            'Sa': 200,
            'R1': 4.8e-4,
            'R2': 3.68e-3,
            'Cin': 4.60e-5,
            'theta_in': 25,
            
            # 电价参数
            'K_buy': 0.8,
            'K_sell': 0.4,
            'K_cur': 2.0,
            'K_delay': 0.05,
            
            # 时间参数（表2）
            'H_A': 6,
            't_e_B1': 17,
            't_l_B1': 32,
            't_e_B2': 18,
            't_l_B2': 32,
            'C_range': [1, 96],
        }
        
        N_servers = params['N_racks'] * params['N_servers_per_rack']
        params['D_dc_max'] = N_servers * params['N_CPU_per_server'] * params['D_cpu_max']
        
        return params