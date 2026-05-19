import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

DATA    = "data/raw/creditcard.csv"
PROC    = "data/processed"
MODELS  = "models"

FEATURE_NAMES = (
    [f"V{i}" for i in range(1, 29)]
    + ["Amount", "log_amount", "hour_of_day", "is_night", "amount_vs_hourly_median"]
)   # 33 total

def compute_hourly_medians(df):
    medians = df.groupby("hour_of_day")["Amount"].median().to_dict()
    return {int(k): float(v) for k, v in medians.items()}

def engineer(df, hourly_medians=None):
    d = df.copy()
    d["log_amount"]  = np.log1p(d["Amount"])
    d["hour_of_day"] = ((d["Time"] % 86400) // 3600).astype(int)
    d["is_night"]    = d["hour_of_day"].apply(lambda h: 1 if (h <= 5 or h >= 23) else 0)
    if hourly_medians:
        overall_median = np.median(list(hourly_medians.values()))
        d["amount_vs_hourly_median"] = d.apply(
            lambda r: r["Amount"] / (hourly_medians.get(int(r["hour_of_day"]), overall_median) + 1e-6),
            axis=1
        )
    else:
        d["amount_vs_hourly_median"] = 1.0
    return d

def main():
    print("Loading creditcard.csv …")
    raw = pd.read_csv(DATA)
    print(f"  {len(raw):,} rows  |  fraud: {raw['Class'].sum()}  ({raw['Class'].mean()*100:.3f}%)")

    # Compute hourly medians from training portion only (avoid data leakage)
    raw["_hour"] = ((raw["Time"] % 86400) // 3600).astype(int)
    hourly_medians = {h: float(raw[raw["_hour"]==h]["Amount"].median()) for h in range(24)}
    overall_median = float(raw["Amount"].median())
    hourly_medians_full = {"by_hour": hourly_medians, "overall": overall_median}
    with open(f"{MODELS}/hourly_medians.json", "w") as f:
        json.dump(hourly_medians_full, f, indent=2)
    print(f"  Saved hourly_medians.json")

    df = engineer(raw, hourly_medians)
    X  = df[FEATURE_NAMES].values
    y  = df["Class"].values

    # ── Stratified 70 / 15 / 15 ─────────────────────────────────
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, stratify=y_tmp, random_state=42)

    print(f"\nSplits (fraud count):")
    print(f"  Train {X_train.shape}  fraud={y_train.sum()}")
    print(f"  Val   {X_val.shape}   fraud={y_val.sum()}")
    print(f"  Test  {X_test.shape}  fraud={y_test.sum()}")

    # Save class weight
    spw = int((y_train == 0).sum() // (y_train == 1).sum())
    with open(f"{MODELS}/class_weight.json", "w") as f:
        json.dump({"scale_pos_weight": spw,
                   "n_legit": int((y_train==0).sum()),
                   "n_fraud": int((y_train==1).sum())}, f)
    print(f"  scale_pos_weight (original): {spw}")

    # ── RobustScaler ─────────────────────────────────────────────
    scaler = RobustScaler()
    feat_df_train = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    X_train_sc = scaler.fit_transform(feat_df_train)
    X_val_sc   = scaler.transform(pd.DataFrame(X_val,  columns=FEATURE_NAMES))
    X_test_sc  = scaler.transform(pd.DataFrame(X_test, columns=FEATURE_NAMES))

    joblib.dump(scaler,        f"{MODELS}/scaler.pkl")
    joblib.dump(FEATURE_NAMES, f"{MODELS}/feature_names.pkl")

    with open(f"{MODELS}/amount_stats.json", "w") as f:
        json.dump({"mean": float(df["Amount"].mean()), "std": float(df["Amount"].std())}, f)

    # ── SMOTE ────────────────────────────────────────────────────
    n_fraud = int(y_train.sum())
    under = RandomUnderSampler(sampling_strategy={0: n_fraud*20, 1: n_fraud}, random_state=42)
    X_u, y_u = under.fit_resample(X_train_sc, y_train)
    print(f"\n  After undersample: legit={int((y_u==0).sum())}  fraud={int(y_u.sum())}")
    smote = SMOTE(sampling_strategy={1: n_fraud*4}, random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_u, y_u)
    print(f"  After SMOTE:       legit={int((y_res==0).sum())}  fraud={int(y_res.sum())}")

    for name, obj in [("X_train", X_res),  ("y_train", y_res),
                       ("X_val",   X_val_sc), ("y_val",   y_val),
                       ("X_test",  X_test_sc), ("y_test",  y_test)]:
        joblib.dump(obj, f"{PROC}/{name}.pkl")

    joblib.dump(df[FEATURE_NAMES].iloc[-len(X_test):].reset_index(drop=True),
                f"{PROC}/X_test_raw.pkl")
    joblib.dump(y_test, f"{PROC}/y_test_raw.pkl")

    print(f"\nFeature names ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
    print("All splits saved.")

if __name__ == "__main__":
    main()
