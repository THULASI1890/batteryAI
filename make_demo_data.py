"""Generate a SYNTHETIC battery-aging dataset shaped like the NASA PCoE data.

Use this only when you don't have the real .mat files. It writes the same
nasa_battery_data.csv that convert_nasa.py produces, so train_models.py and
app.py work unchanged.  Run:  python make_demo_data.py
"""
import numpy as np
import pandas as pd

EOL_CAPACITY = 1.4
rng = np.random.default_rng(42)

# (battery id, start capacity Ah, fade per cycle, number of cycles)
SPECS = [("B0005", 1.86, 0.0030, 168), ("B0006", 2.04, 0.0044, 168),
         ("B0007", 1.89, 0.0028, 168), ("B0018", 1.85, 0.0045, 132)]

frames = []
for name, c0, fade, n in SPECS:
    k = np.arange(1, n + 1)
    # Slightly accelerating fade, measurement noise, and small capacity-regeneration bumps
    cap = c0 - fade * k - 1e-5 * k**1.6 + rng.normal(0, 0.008, n)
    cap += (rng.random(n) < 0.04) * rng.uniform(0.01, 0.04, n)
    cap = np.minimum.accumulate(cap + 0.03) - 0.03 + rng.normal(0, 0.003, n)

    age = (c0 - cap) / c0
    df = pd.DataFrame({
        "battery": name,
        "cycle": k,
        "ambient_temp": 24.0,
        "voltage_mean": 3.53 - 0.25 * age + rng.normal(0, 0.005, n),
        "voltage_min": 2.65 + 0.10 * age + rng.normal(0, 0.02, n),
        "current_mean": 2.0 + rng.normal(0, 0.005, n),
        "temp_mean": 32.0 + 12 * age + rng.normal(0, 0.4, n),
        "temp_max": 36.0 + 16 * age + rng.normal(0, 0.5, n),
        "duration": cap / 2.0 * 3600 + rng.normal(0, 15, n),
        "capacity": cap,
    })
    df["SOH"] = df["capacity"] / df["capacity"].iloc[0] * 100
    below = df.index[df["capacity"] <= EOL_CAPACITY]
    eol = df.loc[below[0], "cycle"] if len(below) else df["cycle"].iloc[-1]
    df["RUL"] = (eol - df["cycle"]).clip(lower=0)
    frames.append(df)

out = pd.concat(frames, ignore_index=True)
out.to_csv("nasa_battery_data.csv", index=False)
print(f"Saved nasa_battery_data.csv with {len(out)} rows (SYNTHETIC demo data)")
