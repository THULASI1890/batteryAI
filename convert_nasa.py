"""Convert NASA PCoE Li-ion battery .mat files (B0005, B0006, B0007, B0018) into one CSV.

Put the .mat files in ./data/ and run:  python convert_nasa.py
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

DATA_DIR = "data"
BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
EOL_CAPACITY = 1.4  # Ah (70% of the 2.0 Ah nominal capacity)


BASE = os.path.dirname(os.path.abspath(__file__))


def find_mat(name):
    """Search the project folder (and sub-folders) for NAME.mat, case-insensitive."""
    for root, _, files in os.walk(BASE):
        for f in files:
            if f.lower() == f"{name}.mat".lower():
                return os.path.join(root, f)
    return None


def extract_battery(name, path):
    mat = loadmat(path, simplify_cells=True)
    cycles = mat[name]["cycle"]
    rows, k = [], 0
    for c in cycles:
        if c["type"] != "discharge":
            continue
        d = c["data"]
        k += 1
        rows.append({
            "battery": name,
            "cycle": k,
            "ambient_temp": float(c["ambient_temperature"]),
            "voltage_mean": float(np.mean(d["Voltage_measured"])),
            "voltage_min": float(np.min(d["Voltage_measured"])),
            "current_mean": float(np.mean(np.abs(d["Current_measured"]))),
            "temp_mean": float(np.mean(d["Temperature_measured"])),
            "temp_max": float(np.max(d["Temperature_measured"])),
            "duration": float(np.max(d["Time"])),
            "capacity": float(np.atleast_1d(d["Capacity"])[0]),
        })
    df = pd.DataFrame(rows)

    # State of Health (%) relative to the first measured capacity
    df["SOH"] = df["capacity"] / df["capacity"].iloc[0] * 100

    # Remaining Useful Life = cycles left until capacity falls to the EOL threshold
    below = df.index[df["capacity"] <= EOL_CAPACITY]
    eol_cycle = df.loc[below[0], "cycle"] if len(below) else df["cycle"].iloc[-1]
    df["RUL"] = (eol_cycle - df["cycle"]).clip(lower=0)
    return df


if __name__ == "__main__":
    frames = []
    for b in BATTERIES:
        path = find_mat(b)
        if path:
            frames.append(extract_battery(b, path))
            print(f"Processed {b} from {path}")
        else:
            print(f"Skipping {b}: {b}.mat not found anywhere under {BASE}")
    if not frames:
        raise SystemExit("\nNo .mat files found. Download the NASA dataset and put "
                         "B0005.mat, B0006.mat, B0007.mat, B0018.mat in the 'data' folder.")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv("nasa_battery_data.csv", index=False)
    print(f"Saved nasa_battery_data.csv with {len(out)} rows")
