# NASA Li-ion Battery Prognostics Project

This repository contains the setup, data loader utilities, and starter scripts for working with the **NASA Prognostics Center of Excellence (PCoE) Li-ion Battery Aging Dataset**.

---

## 📁 Project Structure

```text
nasa_battery_dataset/
├── 5. Battery Data Set/            # Raw NASA .mat dataset files
│   ├── 1. BatteryAgingARC-FY08Q4/  # Benchmark cells: B0005, B0006, B0007, B0018
│   ├── 2. BatteryAgingARC_25_26_...# Additional experimental runs
│   └── ...
├── src/
│   └── nasa_battery/               # Core helper library
│       ├── __init__.py
│       └── loader.py               # MATLAB (.mat) parser & DataFrame extractor
├── notebooks/
│   └── 01_eda_and_baseline.ipynb   # Interactive EDA & baseline ML model
├── demo_eda.py                     # Quick script to verify data & generate plots
├── pyproject.toml                  # Project dependency specification (managed with uv)
├── capacity_degradation.png        # Sample generated capacity degradation curve
└── README.md
```

---

## 🚀 Environment Setup & Activation

The Python virtual environment (`.venv`) is already created and configured with Python 3.11 and all necessary dependencies.

### 1. Activating the Environment
In PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```
In Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

### 2. Launching Jupyter Lab / Notebook
```powershell
.venv\Scripts\jupyter-lab.exe
```
Or open [`notebooks/01_eda_and_baseline.ipynb`](file:///C:/Users/ryan9/.gemini/antigravity/scratch/nasa_battery_dataset/notebooks/01_eda_and_baseline.ipynb) inside your IDE and select the kernel **`Python (NASA Battery .venv)`**.

---

## 💡 Quick Code Example

```python
from nasa_battery import BatteryDataLoader

loader = BatteryDataLoader(data_root="5. Battery Data Set")

# 1. Extract cycle-level tabular summary (capacity, voltage extremes, duration, EIS resistance)
df_summary = loader.extract_cycle_summary("B0005")
print(df_summary.head())

# 2. Extract raw time-series measurements for a specific cycle (voltage, current, temperature vs time)
df_cycle = loader.extract_time_series(battery_id="B0005", cycle_index=2)
print(df_cycle.head())
```

---

## 🎯 Suggested Project Directions

1. **State of Health (SoH) Estimation:**
   - Predict current capacity fade ratio \( \text{SoH} = \frac{C_{\text{current}}}{C_{\text{nominal}}} \) from partial discharge voltage/current curves.
2. **Remaining Useful Life (RUL) Prediction:**
   - Predict how many cycles remain before capacity crosses the 1.4 Ah (30% fade) threshold using LSTMs, Transformers, XGBoost, or State-Space Models (Mamba).
3. **Electrochemical Impedance Spectroscopy (EIS) Analysis:**
   - Correlate electrolyte resistance (\(R_e\)) and charge transfer resistance (\(R_{ct}\)) with cycling age.
4. **Transfer Learning / Cross-Cell Generalization:**
   - Train on standard cells (`B0005`, `B0006`, `B0007`) and test generalization on dynamic/variable load cells (`B0025+` / `B0045+`).

