import numpy as np
import pandas as pd
import joblib
import json
import datetime
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (roc_auc_score, average_precision_score,
    f1_score, fbeta_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve)
from xgboost import XGBClassifier

PROC   = "data/processed"
MODELS = "models"
PLOTS  = "notebooks/plots"

def load():
    return (joblib.load(f"{PROC}/X_train.pkl"), joblib.load(f"{PROC}/y_train.pkl"),
            joblib.load(f"{PROC}/X_val.pkl"),   joblib.load(f"{PROC}/y_val.pkl"),
            joblib.load(f"{PROC}/X_test.pkl"),  joblib.load(f"{PROC}/y_test.pkl"),
            joblib.load(f"{MODELS}/feature_names.pkl"))

def optimal_threshold(y_true, proba):
    best_f2, best_t = 0, 0.5
    best_prec70, best_t70 = 0, 0.5
    for t in np.arange(0.05, 0.80, 0.01):
        yp = (proba >= t).astype(int)
        if yp.sum() == 0: continue
        f2 = fbeta_score(y_true, yp, beta=2)
        pr = precision_score(y_true, yp, zero_division=0)
        rc = recall_score(y_true, yp, zero_division=0)
        if f2 > best_f2:  best_f2, best_t = f2, t
        if pr >= 0.70 and rc > best_prec70: best_prec70, best_t70 = rc, t
    if best_t70 == 0.5: best_t70 = best_t
    return round(best_t, 2), round(best_t70, 2)

def evaluate(name, y_true, proba, threshold):
    yp = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
    fpr = fp / (fp + tn + 1e-9)
    return {"model": name,
            "roc_auc":   round(roc_auc_score(y_true, proba), 4),
            "pr_auc":    round(average_precision_score(y_true, proba), 4),
            "f1":        round(f1_score(y_true, yp), 4),
            "f2":        round(fbeta_score(y_true, yp, beta=2), 4),
            "precision": round(precision_score(y_true, yp, zero_division=0), 4),
            "recall":    round(recall_score(y_true, yp, zero_division=0), 4),
            "fpr":       round(fpr, 4),
            "threshold": threshold,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

# ── plots ──────────────────────────────────────────────────────
def plot_roc(probas, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Random Forest":"#3498DB","XGBoost":"#E74C3C","Isolation Forest":"#F39C12"}
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_test,p):.4f}",
                color=colors.get(name,"gray"), linewidth=2)
    ax.plot([0,1],[0,1],"k--")
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curves — Credit Card Fraud")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{PLOTS}/roc_curves_fraud.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  saved roc_curves_fraud.png")

def plot_pr(probas, y_test, thresh):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Random Forest":"#3498DB","XGBoost":"#E74C3C","Isolation Forest":"#F39C12"}
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        ap = average_precision_score(y_test, p)
        ax.plot(rec, prec, label=f"{name} AP={ap:.4f}", color=colors.get(name,"gray"), linewidth=2)
        if name in thresh:
            t = thresh[name]
            yp = (p >= t).astype(int)
            ax.scatter([recall_score(y_test,yp)], [precision_score(y_test,yp,zero_division=0)],
                       s=120, zorder=5, color=colors.get(name,"gray"), marker="*")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{PLOTS}/pr_curves_fraud.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  saved pr_curves_fraud.png")

def plot_cm(xgb_m, rf_m, y_test, xgb_p, rf_p):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m, p, name in [(axes[0],xgb_m,xgb_p,"XGBoost"),(axes[1],rf_m,rf_p,"Random Forest")]:
        cm = confusion_matrix(y_test, (p>=m["threshold"]).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
        ax.set(title=f"{name} (t={m['threshold']}, Recall={m['recall']:.3f})",
               ylabel="Actual", xlabel="Predicted")
    fig.suptitle("Confusion Matrices at Optimal Threshold", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  saved confusion_matrices.png")

def plot_fi(model, feature_names):
    imp = model.feature_importances_
    idx = np.argsort(imp)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([feature_names[i] for i in idx], imp[idx], color="#E74C3C", alpha=0.8)
    ax.set(title="XGBoost Feature Importance — Top 20", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/feature_importance_fraud.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  saved feature_importance_fraud.png")

def plot_threshold(y_test, proba):
    ts = np.arange(0.05, 0.80, 0.01)
    f2s, precs, recs = [], [], []
    for t in ts:
        yp = (proba >= t).astype(int)
        if yp.sum() == 0: f2s.append(0); precs.append(0); recs.append(0); continue
        f2s.append(fbeta_score(y_test, yp, beta=2))
        precs.append(precision_score(y_test, yp, zero_division=0))
        recs.append(recall_score(y_test, yp, zero_division=0))
    best = ts[np.argmax(f2s)]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(ts, f2s, "#E74C3C", linewidth=2, label="F2")
    ax1.plot(ts, precs, "#3498DB", linewidth=2, linestyle="--", label="Precision")
    ax1.plot(ts, recs, "#27AE60", linewidth=2, linestyle=":", label="Recall")
    ax1.axvline(best, color="#F39C12", linestyle="--", label=f"Best={best:.2f}")
    ax1.set(xlabel="Threshold", ylabel="Score", title="XGBoost: F2 / Precision / Recall vs Threshold")
    ax1.legend(); fig.tight_layout()
    fig.savefig(f"{PLOTS}/threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  saved threshold_analysis.png")

def save_sample_transactions(X_test_raw, y_test, feature_names, proba, threshold):
    """Save 6 real fraud + 6 real legit test transactions for the Try page."""
    fraud_idx = np.where((y_test == 1) & (proba >= threshold))[0]
    legit_idx = np.where((y_test == 0) & (proba < 0.1))[0]

    samples = {"fraud": [], "legit": []}
    for i in fraud_idx[:6]:
        row = {k: round(float(X_test_raw.iloc[i][k]), 4) for k in feature_names}
        row["_risk_pct"] = round(float(proba[i])*100, 1)
        samples["fraud"].append(row)
    for i in legit_idx[:6]:
        row = {k: round(float(X_test_raw.iloc[i][k]), 4) for k in feature_names}
        row["_risk_pct"] = round(float(proba[i])*100, 1)
        samples["legit"].append(row)

    with open(f"{MODELS}/sample_transactions.json", "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  saved {len(samples['fraud'])} fraud + {len(samples['legit'])} legit samples")

def save_shap_summary(model, X_test, feature_names):
    """Mean |SHAP| per feature for the aggregate chart."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test[:500])
        mean_abs = np.abs(sv).mean(axis=0)
        summary = {feature_names[i]: round(float(mean_abs[i]), 5)
                   for i in range(len(feature_names))}
        summary_sorted = dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))
        with open(f"{MODELS}/shap_summary.json", "w") as f:
            json.dump(summary_sorted, f, indent=2)
        joblib.dump(explainer, f"{MODELS}/shap_explainer.pkl")
        print("  saved shap_summary.json + shap_explainer.pkl")
    except Exception as e:
        print(f"  SHAP skipped: {e}")

def main():
    print("Loading data …")
    X_tr, y_tr, X_val, y_val, X_test, y_test, feats = load()
    print(f"  Train {X_tr.shape}  Val {X_val.shape}  Test {X_test.shape}")

    # Class weight info
    with open(f"{MODELS}/class_weight.json") as f:
        cw = json.load(f)
    spw_orig = cw["scale_pos_weight"]
    # After SMOTE (5:1), use resampled ratio for XGB
    spw = max(1, int((y_tr==0).sum() // max((y_tr==1).sum(),1)))
    print(f"  scale_pos_weight (resampled): {spw}")

    # ── Random Forest ──────────────────────────────────────────
    print("\nTraining Random Forest …")
    rf = RandomForestClassifier(n_estimators=300, max_depth=20,
                                 class_weight="balanced", min_samples_leaf=5,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_val_p  = rf.predict_proba(X_val)[:,1]
    rf_test_p = rf.predict_proba(X_test)[:,1]
    rf_t_f2, rf_t_p70 = optimal_threshold(y_val, rf_val_p)
    print(f"  RF optimal threshold: {rf_t_f2}")

    # ── XGBoost ────────────────────────────────────────────────
    print("\nTraining XGBoost …")
    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=spw,
                         eval_metric="aucpr", random_state=42,
                         verbosity=0, early_stopping_rounds=40)
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_p  = xgb.predict_proba(X_val)[:,1]
    xgb_test_p = xgb.predict_proba(X_test)[:,1]
    xgb_t_f2, xgb_t_p70 = optimal_threshold(y_val, xgb_val_p)
    print(f"  XGB optimal threshold: {xgb_t_f2}")

    # ── Isolation Forest ──────────────────────────────────────
    print("\nTraining Isolation Forest …")
    iso = IsolationForest(n_estimators=200,
                           contamination=max(0.001, y_tr.mean()),
                           random_state=42)
    iso.fit(X_tr)
    iso_raw   = iso.predict(X_test)
    iso_score = iso.score_samples(X_test)
    iso_norm  = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-9)
    iso_p     = 1 - iso_norm
    iso_pred  = (iso_raw == -1).astype(int)

    # ── Evaluate ──────────────────────────────────────────────
    xgb_m = evaluate("XGBoost",          y_test, xgb_test_p, xgb_t_f2)
    rf_m  = evaluate("Random Forest",    y_test, rf_test_p,  rf_t_f2)
    iso_m = {"model":"Isolation Forest",
              "roc_auc": round(roc_auc_score(y_test, iso_p), 4),
              "pr_auc":  round(average_precision_score(y_test, iso_p), 4),
              "f1":   round(f1_score(y_test, iso_pred), 4),
              "f2":   round(fbeta_score(y_test, iso_pred, beta=2), 4),
              "precision": round(precision_score(y_test, iso_pred, zero_division=0), 4),
              "recall":    round(recall_score(y_test, iso_pred, zero_division=0), 4),
              "fpr": 0.0, "threshold": 0.5}

    print(f"\n{'Model':<22} {'ROC-AUC':<10} {'PR-AUC':<10} {'F2':<8} {'Recall':<8} {'Prec':<8} {'FPR'}")
    print("-"*72)
    for m in [xgb_m, rf_m, iso_m]:
        print(f"{m['model']:<22} {m['roc_auc']:<10} {m['pr_auc']:<10} "
              f"{m['f2']:<8} {m['recall']:<8} {m['precision']:<8} {m['fpr']}")

    print(f"\nClassification Report (XGBoost):\n"
          + classification_report(y_test, (xgb_test_p>=xgb_t_f2).astype(int),
                                   target_names=["Legitimate","Fraud"]))

    # ── Plots ─────────────────────────────────────────────────
    print("\nGenerating plots …")
    probas = {"XGBoost": xgb_test_p, "Random Forest": rf_test_p, "Isolation Forest": iso_p}
    thresh_map = {"XGBoost": xgb_t_f2, "Random Forest": rf_t_f2}
    plot_roc(probas, y_test)
    plot_pr(probas, y_test, thresh_map)
    plot_cm(xgb_m, rf_m, y_test, xgb_test_p, rf_test_p)
    plot_fi(xgb, feats)
    plot_threshold(y_test, xgb_test_p)

    # ── Save models ───────────────────────────────────────────
    joblib.dump(xgb, f"{MODELS}/xgboost_fraud.pkl")
    joblib.dump(rf,  f"{MODELS}/random_forest_fraud.pkl")

    with open(f"{MODELS}/thresholds.json", "w") as f:
        json.dump({"xgboost_fraud":     {"f2_optimal": xgb_t_f2, "precision_70": xgb_t_p70},
                   "random_forest_fraud":{"f2_optimal": rf_t_f2,  "precision_70": rf_t_p70}}, f, indent=2)

    with open(f"{MODELS}/model_metrics.json", "w") as f:
        json.dump({"xgboost": xgb_m, "random_forest": rf_m, "isolation_forest": iso_m}, f, indent=2)

    with open(f"{MODELS}/best_model.txt", "w") as f:
        f.write("xgboost_fraud")

    with open(f"{MODELS}/last_trained.txt", "w") as f:
        f.write(datetime.datetime.now().isoformat())

    # ── SHAP + samples ────────────────────────────────────────
    print("\nBuilding SHAP explainer …")
    save_shap_summary(xgb, X_test, feats)

    print("\nExtracting sample transactions for Try page …")
    try:
        X_test_raw = joblib.load(f"{PROC}/X_test_raw.pkl")
        save_sample_transactions(X_test_raw, y_test, feats, xgb_test_p, xgb_t_f2)
    except Exception as e:
        print(f"  sample extraction skipped: {e}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
