import os
import sys

# Ensure src is on python path
sys.path.insert(0, os.path.abspath("src"))

import matplotlib.pyplot as plt
import pandas as pd
from nasa_battery import BatteryDataLoader


def main():
    data_dir = "5. Battery Data Set"
    loader = BatteryDataLoader(data_root=data_dir)

    batteries = loader.list_available_batteries()
    print(f"Discovered {len(batteries)} battery datasets across subfolders:")
    print(", ".join(list(batteries.keys())[:12]) + " ...")

    # Focus on the canonical ARC FY08Q4 benchmark batteries
    target_cells = ["B0005", "B0006", "B0007", "B0018"]
    summary_dfs = []

    print("\nExtracting cycle summaries for benchmark cells...")
    for b_id in target_cells:
        if b_id in batteries:
            df = loader.extract_cycle_summary(b_id)
            summary_dfs.append(df)
            discharge_cycles = df[df["type"] == "discharge"]
            print(f"  [{b_id}] Total cycles: {len(df)} | Discharge cycles: {len(discharge_cycles)} | "
                  f"Initial Capacity: {discharge_cycles['capacity'].dropna().iloc[0]:.3f} Ah | "
                  f"Final Capacity: {discharge_cycles['capacity'].dropna().iloc[-1]:.3f} Ah")

    if not summary_dfs:
        print("No summary data found.")
        return

    all_summaries = pd.concat(summary_dfs, ignore_index=True)

    # Filter discharge cycles with valid capacity
    discharge_df = all_summaries[(all_summaries["type"] == "discharge") & (all_summaries["capacity"].notna())].copy()
    
    # Calculate discharge cycle count per battery
    discharge_df["discharge_cycle_num"] = discharge_df.groupby("battery_id").cumcount() + 1

    # Plot capacity degradation
    plt.figure(figsize=(10, 6), dpi=120)
    for b_id in target_cells:
        subset = discharge_df[discharge_df["battery_id"] == b_id]
        plt.plot(subset["discharge_cycle_num"], subset["capacity"], marker="o", markersize=3, label=f"Battery {b_id}")

    # Threshold line (30% fade: 2.0 -> 1.4 Ah)
    plt.axhline(y=1.4, color="red", linestyle="--", alpha=0.7, label="End-of-Life (EOL) Threshold (1.4 Ah)")
    
    plt.title("NASA Li-ion Battery Capacity Degradation Curve", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Discharge Cycle Number", fontsize=12)
    plt.ylabel("Discharge Capacity (Ah)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_plot = "capacity_degradation.png"
    plt.savefig(out_plot)
    print(f"\nSaved capacity fade plot to: {os.path.abspath(out_plot)}")


if __name__ == "__main__":
    main()

