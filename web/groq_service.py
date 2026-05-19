import os
from groq import Groq

MODEL = "llama-3.1-8b-instant"

def _client():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return None
    return Groq(api_key=key)

def is_available():
    return bool(os.environ.get("GROQ_API_KEY", "").strip())

def _call(system_prompt, user_prompt, max_tokens=280):
    client = _client()
    if not client:
        return None
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[AI explanation unavailable: {e}]"

def explain_transaction(risk_pct, recommendation, amount, hour, is_night,
                        factors, threshold, amount_percentile):
    """Plain-English explanation of a single transaction's fraud score."""
    time_ctx = "between 1–5 AM (a high-risk window)" if is_night else f"at {hour}:00"
    pct_ctx  = f"higher than {amount_percentile:.0f}% of all transactions" if amount_percentile else ""

    factor_lines = []
    for f in factors[:4]:
        direction = "pushes risk up" if f.get("direction") == "up" else "pulls risk down"
        if f["feature"] in ("Amount", "log_amount", "amount_zscore"):
            factor_lines.append(f"- Transaction amount (₹{amount:.2f}) — {direction}")
        elif f["feature"] == "hour_of_day":
            factor_lines.append(f"- Hour of transaction ({hour}:00) — {direction}")
        elif f["feature"] == "is_night":
            factor_lines.append(f"- Night-time flag — {direction}")
        else:
            factor_lines.append(
                f"- Behavioral pattern '{f['feature']}' (value {f['value']}) — {direction}")

    system = (
        "You are a fraud analyst AI. Explain credit card fraud detection decisions "
        "in clear, professional English. Be concise (3–4 sentences max). "
        "Never mention V1-V28 or PCA by name — call them 'transaction behavioral patterns'."
    )
    user = f"""A credit card transaction was scored by an XGBoost fraud model (trained on 284,807 real transactions).

- Amount: ₹{amount:.2f} {pct_ctx}
- Time: {time_ctx}
- Risk Score: {risk_pct}%
- Decision: {recommendation} (threshold: {threshold})

Top factors driving this score:
{chr(10).join(factor_lines) or '- No dominant factors identified'}

Explain in 3–4 sentences why this transaction received this risk score. \
Be specific about the amount and time if relevant. End with what an analyst should do."""

    return _call(system, user, max_tokens=220)


def summarize_batch(total, fraud_count, review_count, approve_count,
                    fraud_rate, avg_risk, psi_status, top_rows):
    """Analyst-style summary of a batch upload result."""
    drift_map = {
        "stable":          "consistent with the model's training data",
        "moderate_shift":  "moderately different from training data — monitor closely",
        "major_shift":     "significantly different from training data — model reliability may be reduced",
    }
    drift_text = drift_map.get(psi_status, "unknown distribution")

    top_lines = "\n".join(
        f"  - Row {r['row']}: ₹{r['amount']:.2f} at {r['hour']}:00 — {r['risk_pct']}% risk"
        for r in top_rows[:5]
    ) or "  (none)"

    system = (
        "You are a senior fraud analyst AI. "
        "Summarize batch transaction screening results in a professional, concise style. "
        "3–4 sentences. Focus on what matters: overall risk, patterns, and next action."
    )
    user = f"""Batch fraud screening results ({total} transactions):

- BLOCK (high risk): {fraud_count} transactions ({fraud_rate:.1f}%)
- REVIEW (medium risk): {review_count} ({review_count/max(total,1)*100:.1f}%)
- APPROVE (low risk): {approve_count} ({approve_count/max(total,1)*100:.1f}%)
- Average risk score: {avg_risk:.1f}%
- Data distribution vs training: {drift_text}

Top high-risk transactions:
{top_lines}

Write a 3–4 sentence analyst summary. Highlight the overall risk level, \
note any patterns in the flagged transactions, and recommend next steps."""

    return _call(system, user, max_tokens=260)
