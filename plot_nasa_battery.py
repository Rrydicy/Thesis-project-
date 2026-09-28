import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

data_dir = r"D:\Thesis\dataset"
battery_ids = ['B0005', 'B0006', 'B0007', 'B0018']
colors = {'B0005': '#1f77b4', 'B0006': '#ff7f0e', 'B0007': '#2ca02c', 'B0018': '#d62728'}

def extract_battery_data(mat_path):
    mat = scipy.io.loadmat(mat_path)
    b_name = os.path.splitext(os.path.basename(mat_path))[0]
    cycles_data = mat[b_name][0, 0]['cycle'][0]
    
    discharge_capacities = []
    discharge_cycles = []
    voltage_curves = {} # cycle_idx -> (time, voltage, temp)
    
    d_count = 0
    for c in cycles_data:
        op_type = c['type'][0]
        if op_type == 'discharge':
            d_count += 1
            data = c['data']
            cap = data['Capacity'][0, 0][0, 0] if 'Capacity' in data.dtype.names and data['Capacity'][0, 0].size > 0 else None
            if cap is not None:
                discharge_capacities.append(cap)
                discharge_cycles.append(d_count)
            
            # Save voltage, temp curves for specific cycles
            if d_count in [1, 30, 60, 100, 130]:
                t = data['Time'][0, 0].flatten()
                v = data['Voltage_measured'][0, 0].flatten()
                temp = data['Temperature_measured'][0, 0].flatten()
                voltage_curves[d_count] = (t, v, temp)
                
    return discharge_cycles, discharge_capacities, voltage_curves

print("Extracting NASA Battery data...")
battery_results = {}
for b_id in battery_ids:
    mat_path = os.path.join(data_dir, f"{b_id}.mat")
    if os.path.exists(mat_path):
        cycles, caps, v_curves = extract_battery_data(mat_path)
        battery_results[b_id] = {
            'cycles': np.array(cycles),
            'capacity': np.array(caps),
            'v_curves': v_curves
        }
        print(f"Loaded {b_id}: {len(caps)} discharge cycles (Initial: {caps[0]:.3f} Ah -> Final: {caps[-1]:.3f} Ah)")

# Create 2x2 Multi-Panel Figure
fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

# Panel 1: Capacity Fade vs Cycle
ax1 = axes[0, 0]
for b_id, res in battery_results.items():
    ax1.plot(res['cycles'], res['capacity'], label=f"{b_id} (Cutoff: {'2.7V' if b_id=='B0005' else '2.2V' if b_id=='B0007' else '2.5V'})", 
             color=colors[b_id], linewidth=2, alpha=0.9)

ax1.axhline(y=1.40, color='red', linestyle='--', linewidth=1.8, label='EOL Threshold (1.40 Ah / 70% SoH)')
ax1.axhline(y=2.00, color='gray', linestyle=':', linewidth=1.2, label='Nominal Rating (2.00 Ah)')

# Annotation for capacity recovery
ax1.annotate('Capacity Regeneration\n(Rest-induced Ion Relaxation)', 
             xy=(35, 1.83), xytext=(50, 1.92),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', alpha=0.8),
             fontsize=9)

ax1.set_title('(a) NASA 18650 Battery Capacity Fade & EOL Failure Line', fontweight='bold')
ax1.set_xlabel('Discharge Cycle Number')
ax1.set_ylabel('Discharge Capacity (Ah)')
ax1.set_ylim(1.2, 2.15)
ax1.legend(loc='lower left', frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)

# Panel 2: State of Health (SoH %) Degradation
ax2 = axes[0, 1]
for b_id, res in battery_results.items():
    soh = (res['capacity'] / 2.00) * 100.0
    ax2.plot(res['cycles'], soh, label=b_id, color=colors[b_id], linewidth=2)

ax2.axhline(y=70.0, color='red', linestyle='--', linewidth=1.8, label='EOL Threshold (70% SoH)')
ax2.set_title('(b) State of Health (SoH %) Trajectory', fontweight='bold')
ax2.set_xlabel('Discharge Cycle Number')
ax2.set_ylabel('State of Health (%)')
ax2.set_ylim(60, 105)
ax2.legend(loc='lower left', frameon=True)
ax2.grid(True, linestyle='--', alpha=0.6)

# Panel 3: Voltage Profile Contraction (B0005)
ax3 = axes[1, 0]
v_curves_b5 = battery_results['B0005']['v_curves']
curve_colors = plt.cm.viridis(np.linspace(0, 0.9, len(v_curves_b5)))
for idx, (cyc, (t, v, temp)) in enumerate(sorted(v_curves_b5.items())):
    ax3.plot(t, v, label=f'Cycle {cyc}', color=curve_colors[idx], linewidth=1.8)

ax3.set_title('(c) Discharge Voltage Profiles V(t) Across Aging (B0005)', fontweight='bold')
ax3.set_xlabel('Discharge Time (seconds)')
ax3.set_ylabel('Terminal Voltage (V)')
ax3.axhline(y=2.7, color='gray', linestyle=':', label='Cutoff Voltage (2.7V)')
ax3.legend(loc='upper right', frameon=True)
ax3.grid(True, linestyle='--', alpha=0.6)

# Panel 4: Temperature Rise Dynamics (B0005)
ax4 = axes[1, 1]
for idx, (cyc, (t, v, temp)) in enumerate(sorted(v_curves_b5.items())):
    ax4.plot(t, temp, label=f'Cycle {cyc}', color=curve_colors[idx], linewidth=1.8)

ax4.set_title('(d) Surface Temperature Elevation T(t) During Discharge (B0005)', fontweight='bold')
ax4.set_xlabel('Discharge Time (seconds)')
ax4.set_ylabel('Temperature (°C)')
ax4.legend(loc='upper left', frameon=True)
ax4.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
out_png = r"D:\Thesis\nasa_battery_visualization.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Visualization successfully saved to: {out_png}")
