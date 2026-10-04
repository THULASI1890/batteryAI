"""Train SOH (Random Forest) and RUL (XGBoost) models.  Run: python train_models.py"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

FEATURES = ["cycle", "ambient_temp", "voltage_mean", "voltage_min",
            "current_mean", "temp_mean", "temp_max", "duration"]

df = pd.read_csv("nasa_battery_data.csv")


def make_soh():
    return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)


def make_rul():
    return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                        subsample=0.9, random_state=42)


# Honest evaluation: hold out one entire battery the model has never seen
test_id = "B0018" if "B0018" in df["battery"].values else df["battery"].iloc[-1]
train, test = df[df.battery != test_id], df[df.battery == test_id]
for target, factory in [("SOH", make_soh), ("RUL", make_rul)]:
    m = factory().fit(train[FEATURES], train[target])
    p = m.predict(test[FEATURES])
    print(f"{target} on unseen {test_id}: MAE={mean_absolute_error(test[target], p):.2f}  "
          f"R2={r2_score(test[target], p):.3f}")

# Final models trained on all batteries
joblib.dump(make_soh().fit(df[FEATURES], df["SOH"]), "soh_model.pkl")
joblib.dump(make_rul().fit(df[FEATURES], df["RUL"]), "rul_model.pkl")
print("Saved soh_model.pkl and rul_model.pkl")
