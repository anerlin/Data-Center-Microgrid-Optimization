# Data Center Microgrid Optimization

Replication of IEEE paper: *Optimal Energy Management of Data Center Micro-Grid Considering Computing Workloads Shift*

## Results

| Scenario | Cost (¥) |
|----------|----------|
| Simple Scheduling (grid only) | 6912.00 |
| Optimal Scheduling (with renewables + V2G) | 4218.23 |
| **Cost Reduction** | **39.0%** |

## Data Source

Workload curves extracted from paper Figure 2 using Engauge Digitizer.

## How to Run

```bash
pip install numpy pandas pulp
python run_me.py
