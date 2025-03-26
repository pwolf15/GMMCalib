import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory with multiple CSVs
results_dir = "./results"
csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]

# Create a larger figure to hold subplots for each CSV (2 plots per CSV)
n_files = len(csv_files)
fig, axes = plt.subplots(n_files, 2, figsize=(14, 6 * n_files))

# If there's only one file, axes will not be 2D, so wrap it
if n_files == 1:
    axes = [axes]

# Process each CSV file
for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(os.path.join(results_dir, csv_file))

    # Create columns for absolute error
    df["abs_phi"] = df["diffRollRad"]
    df["abs_theta"] = df["diffPitchRad"]
    df["abs_psi"] = df["diffYawRad"]
    df["abs_tx"] = df["diffTx"]
    df["abs_ty"] = df["diffTy"]
    df["abs_tz"] = df["diffTz"]

    # Euler angles plot
    axes[i][0].boxplot([df["abs_phi"], df["abs_theta"], df["abs_psi"]], labels=["Δφ (Roll)", "Δθ (Pitch)", "Δψ (Yaw)"])
    axes[i][0].set_title(f"{csv_file} - Euler Angle Errors")
    axes[i][0].set_ylabel("Angle error [rad]")
    axes[i][0].grid(True)

    # Translation plot
    axes[i][1].boxplot([df["abs_tx"], df["abs_ty"], df["abs_tz"]], labels=["Δx", "Δy", "Δz"])
    axes[i][1].set_title(f"{csv_file} - Translation Errors")
    axes[i][1].set_ylabel("Translation Error [m]")
    axes[i][1].grid(True)

plt.tight_layout()
output_path = "./combined_gmmcalib_error_plot.png"
plt.savefig(output_path)
output_path
