import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

sns.set_style("darkgrid")
DATA   = "data/raw/creditcard.csv"
OUTDIR = "models"

def load():
    df = pd.read_csv(DATA)
    df["log_amount"]  = np.log1p(df["Amount"])
    df["hour_of_day"] = ((df["Time"] % 86400) // 3600).astype(int)
    df["is_night"]    = df["hour_of_day"].apply(lambda h: 1 if (h<=5 or h>=23) else 0)
    return df

# ── helpers ──────────────────────────────────────────────────
def save(fig, name):
    fig.savefig(f"{OUTDIR}/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")

# ── plots ─────────────────────────────────────────────────────
def plot_class_imbalance(df):
    counts = df["Class"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Legitimate", "Fraudulent"], counts.values,
                  color=["#27AE60", "#E74C3C"], width=0.5, edgecolor="white")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x()+b.get_width()/2, v+500, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution — Real Credit Card Data", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    save(fig, "class_imbalance.png")

def plot_fraud_by_hour(df):
    h = df.groupby("hour_of_day")["Class"].agg(["mean","count"]).reset_index()
    h["pct"] = h["mean"] * 100
    avg = h["pct"].mean()
    colors = ["#E74C3C" if p > avg*2 else "#F39C12" if p > avg else "#27AE60" for p in h["pct"]]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(h["hour_of_day"], h["pct"], color=colors, width=0.7, edgecolor="white")
    ax.axhline(avg, color="white", linewidth=1.5, linestyle="--", label=f"Avg {avg:.3f}%")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Hour of Day", fontsize=13, fontweight="bold")
    ax.set_xticks(range(24)); ax.legend()
    save(fig, "fraud_by_hour.png")

def plot_amount_dist(df):
    legit = df[df["Class"]==0]["log_amount"].sample(5000, random_state=1)
    fraud = df[df["Class"]==1]["log_amount"]
    fig, ax = plt.subplots(figsize=(10, 5))
    legit.plot(kind="kde", ax=ax, color="#27AE60", linewidth=2.5, label="Legitimate")
    fraud.plot(kind="kde", ax=ax, color="#E74C3C", linewidth=2.5, label="Fraudulent")
    ax.fill_between(ax.lines[0].get_xdata(), ax.lines[0].get_ydata(), alpha=0.15, color="#27AE60")
    ax.fill_between(ax.lines[1].get_xdata(), ax.lines[1].get_ydata(), alpha=0.15, color="#E74C3C")
    ax.set_xlabel("log(Amount + 1)"); ax.set_ylabel("Density")
    ax.set_title("Transaction Amount Distribution: Fraud vs Legitimate", fontsize=13, fontweight="bold")
    ax.legend()
    save(fig, "amount_distribution.png")

def plot_top_v_features(df):
    # V14, V10, V12, V4, V11, V17 are typically most discriminative
    top = ["V14", "V10", "V12", "V4", "V11", "V17"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feat in zip(axes.flat, top):
        f_data = df[df["Class"]==1][feat].clip(
            df[feat].quantile(0.01), df[feat].quantile(0.99))
        l_data = df[df["Class"]==0][feat].sample(1000, random_state=1).clip(
            df[feat].quantile(0.01), df[feat].quantile(0.99))
        ax.boxplot([l_data, f_data], labels=["Legit","Fraud"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#3498DB", alpha=0.6),
                   medianprops=dict(color="white", linewidth=2))
        ax.set_title(feat, fontweight="bold")
    fig.suptitle("Top Discriminative PCA Features: Fraud vs Legitimate",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "top_v_features.png")

def plot_v14_dist(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    df[df["Class"]==0]["V14"].clip(-10, 5).sample(5000, random_state=1).plot(
        kind="kde", ax=ax, color="#27AE60", linewidth=2.5, label="Legitimate")
    df[df["Class"]==1]["V14"].clip(-10, 5).plot(
        kind="kde", ax=ax, color="#E74C3C", linewidth=2.5, label="Fraudulent")
    ax.set_title("V14 Distribution — Most Discriminative Feature", fontsize=13, fontweight="bold")
    ax.set_xlabel("V14 value"); ax.legend()
    save(fig, "v14_distribution.png")

def plot_daily_trend(df):
    df2 = df.copy()
    df2["day"] = (df2["Time"] // 86400).astype(int)
    daily = df2.groupby("day")["Class"].agg(n="count", f="sum").reset_index()
    daily["rate"] = daily["f"] / daily["n"] * 100
    daily["roll"] = daily["rate"].rolling(3, min_periods=1).mean()
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()
    ax1.bar(daily["day"], daily["f"], color="#E74C3C", alpha=0.5, label="Fraud Count")
    ax2.plot(daily["day"], daily["roll"], color="#F39C12", linewidth=2, label="Rolling Rate %")
    ax1.set_xlabel("Day"); ax1.set_ylabel("Fraud Count", color="#E74C3C")
    ax2.set_ylabel("Fraud Rate %", color="#F39C12")
    ax1.set_title("2-Day Fraud Trend", fontsize=13, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper left")
    save(fig, "daily_fraud_trend.png")

def plot_amount_scatter(df):
    sample = df.sample(5000, random_state=42)
    fig, ax = plt.subplots(figsize=(12, 5))
    legit = sample[sample["Class"]==0]
    fraud = sample[sample["Class"]==1]
    ax.scatter(legit["hour_of_day"]+np.random.uniform(-0.4,0.4,len(legit)),
               legit["log_amount"], alpha=0.3, c="#27AE60", s=8, label="Legitimate")
    ax.scatter(fraud["hour_of_day"]+np.random.uniform(-0.4,0.4,len(fraud)),
               fraud["log_amount"], alpha=0.9, c="#E74C3C", s=20, zorder=5, label="Fraudulent")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("log(Amount + 1)")
    ax.set_title("Fraud Distribution: Hour vs Amount", fontsize=13, fontweight="bold")
    ax.set_xticks(range(24)); ax.legend()
    save(fig, "hour_vs_amount.png")

def save_stats(df):
    h = df.groupby("hour_of_day")["Class"].mean()
    day = (df["Time"] // 86400).astype(int)
    df2 = df.copy(); df2["day"] = day

    stats = {
        "overall_fraud_rate":    round(df["Class"].mean()*100, 4),
        "total_transactions":    int(len(df)),
        "total_fraud_count":     int(df["Class"].sum()),
        "avg_fraud_amount":      round(df[df["Class"]==1]["Amount"].mean(), 2),
        "avg_legit_amount":      round(df[df["Class"]==0]["Amount"].mean(), 2),
        "top_fraud_hour":        int(h.idxmax()),
        "fraud_rate_night":      round(df[df["is_night"]==1]["Class"].mean()*100, 4),
        "fraud_rate_high_amount":round(df[df["Amount"]>df["Amount"].quantile(0.95)]["Class"].mean()*100, 4),
        "v14_mean_fraud":        round(df[df["Class"]==1]["V14"].mean(), 4),
        "v14_mean_legit":        round(df[df["Class"]==0]["V14"].mean(), 4),
        "v10_mean_fraud":        round(df[df["Class"]==1]["V10"].mean(), 4),
        "v12_mean_fraud":        round(df[df["Class"]==1]["V12"].mean(), 4),
        "peak_fraud_day":        0,
    }
    with open(f"{OUTDIR}/fraud_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\n  Fraud Stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    return stats

def main():
    print("Loading data …")
    df = load()
    print(f"  {len(df):,} rows, {df['Class'].sum()} fraud ({df['Class'].mean()*100:.3f}%)")
    print("\nGenerating plots …")
    plot_class_imbalance(df)
    plot_fraud_by_hour(df)
    plot_amount_dist(df)
    plot_top_v_features(df)
    plot_v14_dist(df)
    plot_daily_trend(df)
    plot_amount_scatter(df)
    save_stats(df)
    print("\nDone — all plots saved to notebooks/plots/")

if __name__ == "__main__":
    main()
