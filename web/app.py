import os, sys, json, io, time, datetime
from functools import wraps
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
import bcrypt
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import (Flask, render_template, request, session,
                   redirect, url_for, jsonify)
from werkzeug.utils import secure_filename
import groq_service

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fraud_ops_2024_xgb")

MODELS   = os.path.join(BASE, "models")
PLOTS    = os.path.join(BASE, "models")   # fraud_stats.json lives here
DATA_CSV = os.path.join(BASE, "data/raw/creditcard.csv")
USERS_F  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_NAMES = V_COLS + ["Amount", "log_amount", "hour_of_day", "is_night", "amount_vs_hourly_median"]

# ── model cache ──────────────────────────────────────────────
_arts = {}
def arts():
    if not _arts:
        _arts["model"]    = joblib.load(os.path.join(MODELS, "xgboost_fraud.pkl"))
        _arts["scaler"]   = joblib.load(os.path.join(MODELS, "scaler.pkl"))
        _arts["features"] = joblib.load(os.path.join(MODELS, "feature_names.pkl"))
        with open(os.path.join(MODELS, "thresholds.json")) as f:
            _arts["thresh"] = json.load(f)
        with open(os.path.join(MODELS, "amount_stats.json")) as f:
            _arts["amt_stats"] = json.load(f)
        try:
            _arts["explainer"] = joblib.load(os.path.join(MODELS, "shap_explainer.pkl"))
        except Exception:
            _arts["explainer"] = None
        try:
            with open(os.path.join(MODELS, "shap_summary.json")) as f:
                _arts["shap_summary"] = json.load(f)
        except Exception:
            _arts["shap_summary"] = {}
        try:
            with open(os.path.join(MODELS, "hourly_medians.json")) as f:
                hm = json.load(f)
            _arts["hourly_medians"] = {int(k): v for k, v in hm["by_hour"].items()}
            _arts["overall_median"] = hm["overall"]
        except Exception:
            _arts["hourly_medians"] = {}
            _arts["overall_median"] = 22.0
    return _arts

def jload(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {}

# ── data cache ────────────────────────────────────────────────
_df_cache = None
def get_df():
    global _df_cache
    if _df_cache is None and os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
        df["log_amount"]  = np.log1p(df["Amount"])
        df["hour_of_day"] = ((df["Time"] % 86400) // 3600).astype(int)
        df["is_night"]    = df["hour_of_day"].apply(lambda h: 1 if h<=5 or h>=23 else 0)
        _df_cache = df
    return _df_cache

# ── auth ──────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def g(*a, **kw):
        if "user" not in session: return redirect(url_for("login"))
        return f(*a, **kw)
    return g

def load_users():
    d = jload(USERS_F)
    return {u["username"]: u for u in d.get("users", [])}

def save_user(username, password, full_name, department, role="viewer"):
    d = jload(USERS_F)
    h = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    d["users"].append({"username": username, "password_hash": h,
                        "role": role, "full_name": full_name, "department": department})
    with open(USERS_F, "w") as f: json.dump(d, f, indent=2)

# ── feature engineering ───────────────────────────────────────
def engineer_row(d, feature_names):
    """Works for both API single-transaction and batch upload rows."""
    amount   = float(d.get("Amount", d.get("amount", 0)))
    time_val = float(d.get("Time",   d.get("time",   0)))
    hour     = int(d.get("hour_of_day", int((time_val % 86400) // 3600)))

    log_amount = np.log1p(amount)
    is_night   = 1 if hour <= 5 or hour >= 23 else 0

    a = arts()
    hm = a.get("hourly_medians", {})
    median_for_hour = hm.get(hour, a.get("overall_median", 22.0))
    amount_vs_hourly_median = amount / (median_for_hour + 1e-6)

    row = {f"V{i}": float(d.get(f"V{i}", d.get(f"v{i}", 0.0))) for i in range(1, 29)}
    row.update({"Amount": amount, "log_amount": log_amount,
                "hour_of_day": hour, "is_night": is_night,
                "amount_vs_hourly_median": amount_vs_hourly_median})
    return pd.DataFrame([[row[f] for f in feature_names]], columns=feature_names)

def score_one(d):
    a = arts()
    df_row = engineer_row(d, a["features"])
    scaled = a["scaler"].transform(df_row)
    thresh = a["thresh"]["xgboost_fraud"]["f2_optimal"]
    risk   = float(a["model"].predict_proba(scaled)[0][1])
    level  = "LOW" if risk < 0.3 else "MEDIUM" if risk < 0.6 else "HIGH"
    rec    = "APPROVE" if risk < 0.3 else "REVIEW" if risk < 0.6 else "BLOCK"

    factors = []
    if a["explainer"] is not None:
        try:
            sv = a["explainer"].shap_values(pd.DataFrame(scaled, columns=a["features"]))
            sv = sv[0]
            top = np.argsort(np.abs(sv))[-5:][::-1]
            for i in top:
                factors.append({
                    "feature": a["features"][i],
                    "value":   round(float(df_row.iloc[0][a["features"][i]]), 3),
                    "shap":    round(float(sv[i]), 4),
                    "impact":  "HIGH" if abs(sv[i]) > 0.05 else "MEDIUM",
                    "direction": "up" if sv[i] > 0 else "down",
                })
        except Exception:
            pass

    amt = float(d.get("Amount", d.get("amount", 0)))
    return {"risk": round(risk, 4), "risk_pct": round(risk*100, 1),
            "is_fraud": bool(risk >= thresh), "level": level,
            "recommendation": rec, "threshold": thresh, "factors": factors,
            "amount_percentile": amount_percentile(amt),
            "hour": int(d.get("hour_of_day", int((float(d.get("Time",0)) % 86400) // 3600))),
            "is_night": bool(engineer_row(d, a["features"]).iloc[0]["is_night"])}

# ── PSI drift check ───────────────────────────────────────────
def compute_psi(uploaded_amounts, n_bins=10):
    """Compare uploaded Amount distribution vs training distribution."""
    df = get_df()
    if df is None: return None, "unknown"
    train_ref = df["Amount"].values

    bins = np.percentile(train_ref, np.linspace(0, 100, n_bins+1))
    bins[0], bins[-1] = -np.inf, np.inf

    def bucket(values):
        counts, _ = np.histogram(values, bins=bins)
        pct = counts / max(len(values), 1)
        return np.where(pct == 0, 0.0001, pct)

    ref_pct = bucket(train_ref)
    new_pct = bucket(np.array(uploaded_amounts))
    psi = float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))

    status = "stable" if psi < 0.1 else "moderate_shift" if psi < 0.25 else "major_shift"
    return round(psi, 4), status

# ── chart builders ────────────────────────────────────────────
DARK = dict(plot_bgcolor="#111827", paper_bgcolor="#111827",
            font=dict(color="#e2e8f0", size=12),
            margin=dict(t=45, b=35, l=50, r=20), height=300)

def fig_json(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def chart_fraud_by_hour(df):
    h = df.groupby("hour_of_day")["Class"].mean().reset_index()
    h["pct"] = h["Class"] * 100
    avg = h["pct"].mean()
    colors = ["#E74C3C" if p>avg*2 else "#F39C12" if p>avg else "#27AE60" for p in h["pct"]]
    fig = go.Figure()
    fig.add_bar(x=h["hour_of_day"], y=h["pct"], marker_color=colors,
                hovertemplate="Hour %{x}: %{y:.3f}%<extra></extra>")
    fig.add_hline(y=avg, line_dash="dash", line_color="#94a3b8",
                  annotation_text=f"Avg {avg:.3f}%")
    fig.update_layout(title="Fraud Rate by Hour", xaxis_title="Hour",
                      yaxis_title="Fraud Rate (%)", **DARK)
    return fig_json(fig)

def chart_amount_dist(df):
    legit = np.log1p(df[df["Class"]==0]["Amount"].sample(5000, random_state=1))
    fraud = np.log1p(df[df["Class"]==1]["Amount"])
    fig = go.Figure()
    fig.add_histogram(x=legit, nbinsx=60, name="Legitimate", opacity=0.55,
                      marker_color="#27AE60")
    fig.add_histogram(x=fraud, nbinsx=40, name="Fraudulent", opacity=0.85,
                      marker_color="#E74C3C")
    fig.update_layout(barmode="overlay", title="Amount Distribution (log scale)",
                      xaxis_title="log(Amount+1)", legend=dict(x=0.7,y=0.9), **DARK)
    return fig_json(fig)

def chart_v14_dist(df):
    legit = df[df["Class"]==0]["V14"].clip(-15, 5).sample(5000, random_state=1)
    fraud = df[df["Class"]==1]["V14"].clip(-15, 5)
    fig = go.Figure()
    fig.add_histogram(x=legit, nbinsx=60, name="Legitimate", opacity=0.55,
                      marker_color="#27AE60")
    fig.add_histogram(x=fraud, nbinsx=40, name="Fraudulent", opacity=0.85,
                      marker_color="#E74C3C")
    fig.update_layout(barmode="overlay", title="V14 — Most Discriminative Feature",
                      xaxis_title="V14 value", legend=dict(x=0.7,y=0.9), **DARK)
    return fig_json(fig)

def chart_daily_trend(df):
    df2 = df.copy()
    df2["day"] = (df2["Time"] // 86400).astype(int)
    d = df2.groupby("day")["Class"].agg(n="count", f="sum").reset_index()
    d["rate"] = d["f"] / d["n"] * 100
    d["roll"] = d["rate"].rolling(2, min_periods=1).mean()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=d["day"], y=d["f"], name="Fraud Count",
                marker_color="#E74C3C", opacity=0.5, secondary_y=False)
    fig.add_scatter(x=d["day"], y=d["roll"], name="Rolling Rate %",
                    line=dict(color="#F39C12", width=2.5), secondary_y=True)
    fig.update_layout(title="Daily Fraud Trend", **{**DARK, "height": 300})
    return fig_json(fig)

def chart_confusion(xgb, rf):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["XGBoost","Random Forest"])
    for m, col in [(xgb,1),(rf,2)]:
        tp,fp,tn,fn = m.get("tp",0),m.get("fp",0),m.get("tn",0),m.get("fn",0)
        fig.add_heatmap(z=[[tn,fp],[fn,tp]],
                        text=[[f"TN<br>{tn:,}",f"FP<br>{fp}"],[f"FN<br>{fn}",f"TP<br>{tp}"]],
                        texttemplate="%{text}", colorscale="Blues", showscale=False,
                        x=["Pred Legit","Pred Fraud"], y=["Actual Legit","Actual Fraud"],
                        row=1, col=col)
    fig.update_layout(title="Confusion Matrices", **{**DARK,"height":320})
    return fig_json(fig)

def chart_feature_imp():
    a = arts()
    imp = a["model"].feature_importances_
    idx = np.argsort(imp)[-15:]
    fig = go.Figure(go.Bar(x=imp[idx], y=[a["features"][i] for i in idx],
                           orientation="h", marker_color="#3498DB", opacity=0.85,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(title="Top 15 Feature Importances (XGBoost)", xaxis_title="Importance",
                      **{**DARK,"height":420,"margin":dict(t=45,b=35,l=160,r=20)})
    return fig_json(fig)

def chart_threshold():
    try:
        curve_path = os.path.join(MODELS, "threshold_curve.json")
        with open(curve_path) as f:
            d = json.load(f)
        ts    = d["thresholds"]
        f2s   = d["f2"]
        precs = d["precision"]
        recs  = d["recall"]
        best  = ts[int(np.argmax(f2s))]
        fig = go.Figure()
        fig.add_scatter(x=ts, y=f2s,   name="F2 Score",  line=dict(color="#E74C3C", width=2.5))
        fig.add_scatter(x=ts, y=precs, name="Precision", line=dict(color="#3498DB", width=2, dash="dash"))
        fig.add_scatter(x=ts, y=recs,  name="Recall",    line=dict(color="#27AE60", width=2, dash="dot"))
        fig.add_vline(x=best, line_dash="dash", line_color="#F39C12",
                      annotation_text=f"Best={best:.2f}")
        fig.update_layout(title="F2 / Precision / Recall vs Threshold",
                          xaxis_title="Threshold", legend=dict(x=0.7, y=0.5),
                          **{**DARK, "height": 320})
        return fig_json(fig)
    except Exception:
        return fig_json(go.Figure())

def chart_shap_summary():
    a = arts()
    summary = a.get("shap_summary", {})
    if not summary: return fig_json(go.Figure())
    items = list(summary.items())[:15]
    names  = [x[0] for x in items]
    values = [x[1] for x in items]
    fig = go.Figure(go.Bar(x=values[::-1], y=names[::-1], orientation="h",
                           marker_color="#8E44AD", opacity=0.85,
                           hovertemplate="%{y}: %{x:.5f}<extra></extra>"))
    fig.update_layout(title="Aggregate SHAP — Mean |SHAP| per Feature (500 test samples)",
                      xaxis_title="Mean |SHAP value|",
                      **{**DARK,"height":440,"margin":dict(t=45,b=35,l=160,r=20)})
    return fig_json(fig)

# ── auth routes ───────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("dashboard") if "user" in session else url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if "user" in session: return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","")
        users = load_users()
        if u in users and bcrypt.checkpw(p.encode(), users[u]["password_hash"].encode()):
            session.update({"user": u, "role": users[u]["role"],
                            "full_name": users[u]["full_name"],
                            "department": users[u].get("department","")})
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error, mode="login")

@app.route("/signup", methods=["GET","POST"])
def signup():
    if "user" in session: return redirect(url_for("dashboard"))
    error = None; success = None
    if request.method == "POST":
        username   = request.form.get("username","").strip()
        password   = request.form.get("password","")
        confirm    = request.form.get("confirm","")
        full_name  = request.form.get("full_name","").strip() or username
        department = request.form.get("department","").strip() or "Guest"
        users = load_users()
        if not username or not password:         error = "Username and password required."
        elif username in users:                   error = f"'{username}' is taken."
        elif len(password) < 6:                  error = "Password must be ≥ 6 chars."
        elif password != confirm:                 error = "Passwords do not match."
        else:
            save_user(username, password, full_name, department)
            success = f"Account created! Log in as '{username}'."
    return render_template("login.html", error=error, success=success, mode="signup")

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login"))

# ── page routes ───────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    stats   = jload(os.path.join(PLOTS, "fraud_stats.json"))
    metrics = jload(os.path.join(MODELS, "model_metrics.json"))
    thresh  = jload(os.path.join(MODELS, "thresholds.json"))
    xgb = metrics.get("xgboost", {})
    last_trained = ""
    lt = os.path.join(MODELS, "last_trained.txt")
    if os.path.exists(lt):
        with open(lt) as f: last_trained = f.read().strip()[:16]
    return render_template("dashboard.html", stats=stats, xgb=xgb,
                           thresh=thresh.get("xgboost_fraud",{}),
                           last_trained=last_trained,
                           day_names=["Monday","Tuesday","Wednesday",
                                      "Thursday","Friday","Saturday","Sunday"])

@app.route("/metrics")
@login_required
def metrics():
    m = jload(os.path.join(MODELS, "model_metrics.json"))
    t = jload(os.path.join(MODELS, "thresholds.json"))
    xgb = m.get("xgboost",{});  rf = m.get("random_forest",{});  iso = m.get("isolation_forest",{})
    charts = {"confusion": chart_confusion(xgb, rf),
              "features":  chart_feature_imp(),
              "threshold": chart_threshold(),
              "shap":      chart_shap_summary()}
    df = get_df()
    extra_charts = {}
    if df is not None:
        extra_charts["hour"]   = chart_fraud_by_hour(df)
        extra_charts["amount"] = chart_amount_dist(df)
        extra_charts["v14"]    = chart_v14_dist(df)
        extra_charts["trend"]  = chart_daily_trend(df)
    return render_template("metrics.html", xgb=xgb, rf=rf, iso=iso,
                           thresh=t.get("xgboost_fraud",{}),
                           charts={**charts, **extra_charts})

@app.route("/try")
@login_required
def try_model():
    samples = jload(os.path.join(MODELS, "sample_transactions.json"))
    stats   = jload(os.path.join(PLOTS,  "fraud_stats.json"))
    return render_template("try.html", samples=samples, stats=stats)

@app.route("/upload")
@login_required
def upload():
    return render_template("upload.html")

# ── api routes ────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json(force=True)
    if not data: return jsonify({"error":"No JSON body"}), 400
    try:
        return jsonify(score_one(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
@login_required
def api_upload():
    if "file" not in request.files: return jsonify({"error":"No file"}), 400
    f = request.files["file"]
    try:
        df = pd.read_csv(f) if secure_filename(f.filename).endswith(".csv") else pd.read_excel(f)
    except Exception as e:
        return jsonify({"error": f"Cannot read file: {e}"}), 400

    # Check required columns (V1-V28 + Amount + Time)
    required_v = V_COLS + ["Amount"]
    missing = [c for c in required_v if c not in df.columns]
    if missing: return jsonify({"error": f"Missing columns: {missing}"}), 400

    # Add Time if missing (default to 0)
    if "Time" not in df.columns: df["Time"] = 0

    a = arts()
    feats = a["features"]
    thresh = a["thresh"]["xgboost_fraud"]["f2_optimal"]

    rows = [engineer_row(row.to_dict(), feats) for _, row in df.iterrows()]
    feat_df = pd.concat(rows, ignore_index=True)
    scaled  = a["scaler"].transform(feat_df)
    proba   = a["model"].predict_proba(scaled)[:,1]

    # PSI drift check on Amount
    psi_val, psi_status = compute_psi(df["Amount"].tolist())

    # Velocity: count txns per card in this batch if card_id column present
    has_velocity = "card_id" in df.columns
    card_velocity = {}
    if has_velocity:
        card_velocity = df.groupby("card_id").size().to_dict()

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        risk = float(proba[i])
        rec  = "BLOCK" if risk >= 0.6 else "REVIEW" if risk >= 0.3 else "APPROVE"
        entry = {"row": i+1, "amount": round(float(row["Amount"]),2),
                 "hour": int((float(row.get("Time",0)) % 86400) // 3600),
                 "risk_pct": round(risk*100, 1), "recommendation": rec}
        if has_velocity:
            card = row.get("card_id", "")
            count = int(card_velocity.get(card, 1))
            entry["card_id"]  = str(card)
            entry["card_txn_count"] = count
            entry["velocity_flag"] = count >= 3   # 3+ txns from same card = suspicious
        results.append(entry)

    results_sorted = sorted(results, key=lambda x: x["risk_pct"], reverse=True)
    n = len(results)
    fraud_n  = sum(1 for r in results if r["recommendation"]=="BLOCK")
    review_n = sum(1 for r in results if r["recommendation"]=="REVIEW")
    # High-velocity cards (3+ txns from same card)
    velocity_alerts = []
    if has_velocity:
        suspicious_cards = {c: cnt for c, cnt in card_velocity.items() if cnt >= 3}
        for card, cnt in suspicious_cards.items():
            card_rows = [r for r in results if r.get("card_id") == str(card)]
            avg_risk = round(np.mean([r["risk_pct"] for r in card_rows]), 1)
            velocity_alerts.append({"card_id": str(card), "txn_count": cnt, "avg_risk": avg_risk})

    return jsonify({
        "total": n, "fraud_count": fraud_n, "review_count": review_n,
        "approve_count": n-fraud_n-review_n,
        "fraud_rate": round(fraud_n/n*100, 2) if n else 0,
        "avg_risk": round(float(np.mean(proba))*100, 1),
        "psi": psi_val, "psi_status": psi_status,
        "has_velocity": has_velocity,
        "velocity_alerts": velocity_alerts,
        "results": results_sorted[:1000],
    })

# ── amount percentile helper ──────────────────────────────────
def amount_percentile(amount):
    df = get_df()
    if df is None:
        return None
    return round(float((df["Amount"] < amount).mean() * 100), 1)

# ── groq routes ───────────────────────────────────────────────
@app.route("/api/explain", methods=["POST"])
@login_required
def api_explain():
    d = request.get_json(force=True)
    if not groq_service.is_available():
        return jsonify({"error": "GROQ_API_KEY not set"}), 503
    try:
        text = groq_service.explain_transaction(
            risk_pct        = d.get("risk_pct", 0),
            recommendation  = d.get("recommendation", ""),
            amount          = float(d.get("amount", 0)),
            hour            = int(d.get("hour", 0)),
            is_night        = bool(d.get("is_night", 0)),
            factors         = d.get("factors", []),
            threshold       = d.get("threshold", 0.79),
            amount_percentile = d.get("amount_percentile"),
        )
        return jsonify({"explanation": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/summarize", methods=["POST"])
@login_required
def api_summarize():
    d = request.get_json(force=True)
    if not groq_service.is_available():
        return jsonify({"error": "GROQ_API_KEY not set"}), 503
    try:
        text = groq_service.summarize_batch(
            total         = d.get("total", 0),
            fraud_count   = d.get("fraud_count", 0),
            review_count  = d.get("review_count", 0),
            approve_count = d.get("approve_count", 0),
            fraud_rate    = d.get("fraud_rate", 0),
            avg_risk      = d.get("avg_risk", 0),
            psi_status    = d.get("psi_status", "stable"),
            top_rows      = d.get("top_rows", []),
        )
        return jsonify({"summary": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/groq_status")
@login_required
def api_groq_status():
    return jsonify({"available": groq_service.is_available()})

if __name__ == "__main__":
    app.run(debug=False, port=5001)
