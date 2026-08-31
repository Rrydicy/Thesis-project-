import os
import glob
import scipy.io as sio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class BatteryDataLoader:
    """
    Utility to load and parse NASA Prognostics Center of Excellence (PCoE)
    Li-ion Battery Aging Dataset MATLAB (.mat) files.
    """

    def __init__(self, data_root: str):
        self.data_root = data_root

    def list_available_batteries(self) -> Dict[str, str]:
        """Find all available battery .mat files and their paths."""
        mat_files = glob.glob(os.path.join(self.data_root, "**", "*.mat"), recursive=True)
        batteries = {}
        for f in mat_files:
            battery_id = os.path.splitext(os.path.basename(f))[0]
            batteries[battery_id] = f
        return dict(sorted(batteries.items()))

    def load_battery_mat(self, battery_id: str) -> np.ndarray:
        """Load the raw cycle array from a battery .mat file."""
        batteries = self.list_available_batteries()
        if battery_id not in batteries:
            raise FileNotFoundError(f"Battery ID '{battery_id}' not found in {self.data_root}")
        
        file_path = batteries[battery_id]
        mat = sio.loadmat(file_path)
        if battery_id not in mat:
            valid_keys = [k for k in mat.keys() if not k.startswith("__")]
            if valid_keys:
                battery_id = valid_keys[0]
            else:
                raise KeyError(f"No valid data key found in {file_path}")
        
        return mat[battery_id][0, 0]["cycle"][0]

    def extract_cycle_summary(self, battery_id: str) -> pd.DataFrame:
        """
        Extract high-level metrics for every cycle (charge, discharge, impedance).
        Returns a DataFrame with one row per cycle.
        """
        cycles_raw = self.load_battery_mat(battery_id)
        records = []

        for idx, c in enumerate(cycles_raw):
            c_type = str(c["type"][0])
            amb_temp = float(c["ambient_temperature"][0, 0]) if "ambient_temperature" in c.dtype.names else np.nan
            
            time_raw = c["time"][0] if "time" in c.dtype.names and len(c["time"][0]) == 6 else None
            timestamp = None
            if time_raw is not None:
                try:
                    timestamp = pd.Timestamp(
                        year=int(time_raw[0]),
                        month=int(time_raw[1]),
                        day=int(time_raw[2]),
                        hour=int(time_raw[3]),
                        minute=int(time_raw[4]),
                        second=int(time_raw[5])
                    )
                except Exception:
                    pass

            data_struct = c["data"]
            fields = data_struct.dtype.names or ()
            
            capacity = np.nan
            v_min = np.nan
            v_max = np.nan
            temp_max = np.nan
            duration = np.nan
            re_est = np.nan
            rct_est = np.nan

            if "Capacity" in fields:
                cap_val = data_struct["Capacity"][0, 0]
                if cap_val.size > 0:
                    capacity = float(cap_val.flatten()[0])

            if "Voltage_measured" in fields:
                v = data_struct["Voltage_measured"][0, 0].flatten()
                if v.size > 0:
                    v_min = float(np.min(v))
                    v_max = float(np.max(v))

            if "Temperature_measured" in fields:
                t = data_struct["Temperature_measured"][0, 0].flatten()
                if t.size > 0:
                    temp_max = float(np.max(t))

            if "Time" in fields:
                t_vec = data_struct["Time"][0, 0].flatten()
                if t_vec.size > 0:
                    duration = float(t_vec[-1] - t_vec[0])

            if "Re" in fields:
                re_val = data_struct["Re"][0, 0]
                if re_val.size > 0:
                    re_est = float(re_val.flatten()[0])

            if "Rct" in fields:
                rct_val = data_struct["Rct"][0, 0]
                if rct_val.size > 0:
                    rct_est = float(rct_val.flatten()[0])

            records.append({
                "battery_id": battery_id,
                "cycle_index": idx + 1,
                "type": c_type,
                "timestamp": timestamp,
                "ambient_temperature": amb_temp,
                "capacity": capacity,
                "duration_sec": duration,
                "v_min": v_min,
                "v_max": v_max,
                "temp_max": temp_max,
                "Re": re_est,
                "Rct": rct_est
            })

        return pd.DataFrame(records)

    def extract_time_series(self, battery_id: str, cycle_index: int) -> pd.DataFrame:
        """
        Extract detailed measurement time-series for a specific cycle (1-based index).
        """
        cycles_raw = self.load_battery_mat(battery_id)
        if cycle_index < 1 or cycle_index > len(cycles_raw):
            raise IndexError(f"cycle_index {cycle_index} out of range [1, {len(cycles_raw)}]")

        c = cycles_raw[cycle_index - 1]
        data_struct = c["data"]
        fields = data_struct.dtype.names or ()

        df_dict = {}
        for f in fields:
            arr = data_struct[f][0, 0].flatten()
            if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                df_dict[f] = arr

        lengths = [len(v) for v in df_dict.values()]
        if not lengths:
            return pd.DataFrame()
        
        min_len = min(lengths)
        df = pd.DataFrame({k: v[:min_len] for k, v in df_dict.items()})

        df["battery_id"] = battery_id
        df["cycle_index"] = cycle_index
        df["type"] = str(c["type"][0])
        return df

