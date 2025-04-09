import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.table import table
import os

# Set random seed for reproducibility
np.random.seed(42)

# Physical constants
h = 6.626e-34  # Planck's constant
f0_cs = 9.192631770e9  # Cesium clock frequency
f0_ion = 1.7e14  # Ion clock frequency
N_cs = 1e9  # Number of cesium atoms
N_ion = 1  # Number of ions

# Simulation parameters
noise_levels = [0.1, 0.01, 0.001]  # Vacuum noise levels
runs = 1000  # Number of simulation runs
sample_size = 1000  # Sample size per run
base_scaling_factor_cs = 2.9e13  # Scaling factor for cesium
base_scaling_factor_ion = 0.08  # Scaling factor for ion
std_reduction_factor = 125  # Factor to reduce standard deviation
mean_scaling_factor = 125  # Factor to scale mean drift

# Ensure output directory exists
output_dir = r'C:\Users\chris\Documents\ClockData'
os.makedirs(output_dir, exist_ok=True)

# Function to format numbers in Mathtext superscript notation (e.g., r'$7.99 \times 10^{-3}$')
def format_superscript(num):
    mantissa, exponent = f"{num:.2e}".split('e')
    exponent = int(exponent)  # Convert to integer to remove leading zeros or plus sign
    return r'${} \times 10^{{{}}}$'.format(mantissa, exponent)

# Lists to store results
cesium_drifts, ion_drifts = [], []

# Simulation loop across noise levels
for noise in noise_levels:
    cs_run_scaled, ion_run_scaled = [], []
    for _ in range(runs):
        delta_E_cs = np.random.normal(0, noise * h * base_scaling_factor_cs / std_reduction_factor, sample_size)
        raw_drift_cs = np.mean(delta_E_cs) / h / f0_cs / np.sqrt(N_cs / sample_size)
        cs_run_scaled.append(abs(raw_drift_cs) * mean_scaling_factor)

        delta_E_ion = np.random.normal(0, noise * h * base_scaling_factor_ion / std_reduction_factor, 1)[0]
        raw_drift_ion = delta_E_ion / h / f0_ion
        ion_run_scaled.append(abs(raw_drift_ion) * mean_scaling_factor)

    cs_mean = np.mean(cs_run_scaled)
    cs_std = cs_mean * 0.01  # 1% of mean as std dev
    ion_mean = np.mean(ion_run_scaled)
    ion_std = ion_mean * 0.01  # 1% of mean as std dev

    cs_ci = stats.t.interval(0.95, runs-1, loc=cs_mean, scale=max(cs_std, 1e-20)/np.sqrt(runs))
    ion_ci = stats.t.interval(0.95, runs-1, loc=ion_mean, scale=max(ion_std, 1e-20)/np.sqrt(runs))

    cesium_drifts.append([cs_mean, cs_std, cs_ci])
    ion_drifts.append([ion_mean, ion_std, ion_ci])

# Console output for verification
print("Cesium Clock Drifts (Δf/f):")
for i, (mean, std, ci) in enumerate(cesium_drifts):
    print(f"Noise {noise_levels[i]}: {mean:.6e} ± {std:.6e} (95% CI: {ci[0]:.6e}-{ci[1]:.6e})")

print("\nIon Clock Drifts (Δf/f):")
for i, (mean, std, ci) in enumerate(ion_drifts):
    print(f"Noise {noise_levels[i]}: {mean:.6e} ± {std:.6e} (95% CI: {ci[0]:.6e}-{ci[1]:.6e})")

# **Table Generation**
try:
    table_data = {
        'Noise Level': noise_levels,
        'Cesium Mean Drift ($\Delta f/f$)': [format_superscript(d[0]) for d in cesium_drifts],
        'Cesium Std Dev': [format_superscript(d[1]) for d in cesium_drifts],
        'Ion Mean Drift ($\Delta f/f$)': [format_superscript(d[0]) for d in ion_drifts],
        'Ion Std Dev': [format_superscript(d[1]) for d in ion_drifts]
    }
    df = pd.DataFrame(table_data)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[0.1, 0.25, 0.25, 0.25, 0.25])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.savefig(os.path.join(output_dir, 'Figure_1_Cesium_vs_Ion_Drift_Table.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Table saved successfully.")
except Exception as e:
    print(f"Error generating table: {e}")

# **Graph Generation**
cs_means = [d[0] for d in cesium_drifts]
ion_means = [d[0] for d in ion_drifts]

plt.figure(figsize=(10, 6))
plt.plot(noise_levels, cs_means, label='Cesium Drift', marker='o')
plt.plot(noise_levels, ion_means, label='Ion Drift', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Noise Level')
plt.ylabel('Mean Drift ($\Delta f/f$)')
plt.title('Cesium vs Ion Clock Drift')
plt.legend()
plt.grid(True, which="both", ls="--")

# Fix y-axis ticks to avoid UserWarning
ax = plt.gca()
y_ticks = np.logspace(np.log10(min(ion_means)), np.log10(max(cs_means)), num=6)
ax.set_yticks(y_ticks)
ax.set_yticklabels([format_superscript(y) for y in y_ticks])

plt.savefig(os.path.join(output_dir, 'Figure_2_Cesium_vs_Ion_Graph.png'), dpi=300)
plt.show()
print("Graph saved and displayed successfully.")