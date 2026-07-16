import json, os, numpy as np, joblib
import gradio as gr
from huggingface_hub import hf_hub_download

# ─── Load model from HuggingFace Hub ──────────────────────────────────────────
REPO  = 'nareen99/edumine-xgboost'
TOKEN = os.environ.get('HF_TOKEN')

try:
    model    = joblib.load(hf_hub_download(REPO, 'week3_model.pkl',    token=TOKEN))
    explainer= joblib.load(hf_hub_download(REPO, 'week3_explainer.pkl',token=TOKEN))
    with open(hf_hub_download(REPO, 'week3_features.json',             token=TOKEN)) as f:
        FEATURES = json.load(f)
    print(f"✅ Model loaded from {REPO}")
except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    model, explainer, FEATURES = None, None, []

# ─── Prediction logic ─────────────────────────────────────────────────────────
def predict_api(total_clicks, active_days, mean_daily_clicks, activity_diversity,
                weeks_active, avg_score, num_submissions, early_clicks,
                edu_level, studied_credits, num_of_prev_attempts):
    if model is None:
        return {"error": "Model not loaded. Upload model files to HuggingFace."}

    feat_map = dict(
        total_clicks=total_clicks,        active_days=active_days,
        mean_daily_clicks=mean_daily_clicks, activity_diversity=activity_diversity,
        weeks_active=weeks_active,        click_std=5.0,
        max_weekly=0.0,                   min_weekly=0.0,
        num_submissions=num_submissions,  avg_score=avg_score,
        min_score=avg_score * 0.8,        early_clicks=early_clicks,
        gender_M=0,                       disability_Y=0,
        edu_level=edu_level,              num_of_prev_attempts=num_of_prev_attempts,
        studied_credits=studied_credits,
    )

    X         = np.array([[feat_map[f] for f in FEATURES]])
    prob      = float(model.predict_proba(X)[0, 1])
    risk_pct  = round(prob * 100, 1)
    risk_lbl  = 'High Risk' if prob > 0.6 else 'Moderate Risk' if prob > 0.4 else 'Low Risk'
    shap_vals = explainer.shap_values(X)[0]
    top_factors = sorted(zip(FEATURES, shap_vals.tolist()), key=lambda x: -abs(x[1]))[:5]

    return {
        'risk_probability': prob,
        'risk_percent':     risk_pct,
        'risk_label':       risk_lbl,
        'top_factors':      [{'feature': f, 'shap': round(s, 4)} for f, s in top_factors],
    }

# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="EduMine — At-Risk Student Predictor") as demo:
    gr.Markdown("## 📊 EduMine — At-Risk Student Early Warning System")
    gr.Markdown("Predict dropout risk from first 3 weeks of LMS engagement. "
                "XGBoost trained on 32,593 OULAD students with SHAP explainability.")

    with gr.Row():
        with gr.Column():
            total_clicks    = gr.Number(label="Total Clicks",           value=40)
            active_days     = gr.Number(label="Active Days",            value=10)
            mean_daily      = gr.Number(label="Avg Daily Clicks",       value=4)
            diversity       = gr.Number(label="Activity Types Used",    value=3)
            weeks_active    = gr.Number(label="Weeks Active",           value=3)
            avg_score       = gr.Number(label="Avg Assessment Score",   value=55)
            num_submissions = gr.Number(label="Assessments Submitted",  value=1)
            early_clicks    = gr.Number(label="Early Clicks (Wk 1–2)", value=15)
            edu_level       = gr.Slider(0, 4, step=1, label="Education Level (0–4)", value=2)
            credits         = gr.Number(label="Credits Enrolled",       value=60)
            prev_attempts   = gr.Number(label="Previous Attempts",      value=0)
            btn = gr.Button("Predict Risk →", variant="primary")
        with gr.Column():
            result_out = gr.JSON(label="Prediction Result")

    btn.click(
        predict_api,
        inputs=[total_clicks, active_days, mean_daily, diversity, weeks_active,
                avg_score, num_submissions, early_clicks, edu_level, credits, prev_attempts],
        outputs=result_out,
        api_name="predict",   # exposes POST /run/predict for the React frontend
    )

# ─── Launch (keeps the Space alive — DO NOT remove) ──────────────────────────
demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
