import json
import numpy as np
import joblib
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# ─── Load model from HuggingFace Hub ──────────────────────────────────────────
import os
REPO  = 'nareen99/edumine-xgboost'
TOKEN = os.environ.get('HF_TOKEN')       # set this as a Space secret

try:
    model    = joblib.load(hf_hub_download(REPO, 'week3_model.pkl',    token=TOKEN))
    explainer= joblib.load(hf_hub_download(REPO, 'week3_explainer.pkl',token=TOKEN))
    with open(hf_hub_download(REPO, 'week3_features.json',             token=TOKEN)) as f:
        FEATURES = json.load(f)
    print(f"✅ Model loaded from {REPO}")
except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    model, explainer, FEATURES = None, None, []

# ─── FastAPI app (REST API for the React frontend) ────────────────────────────
rest = FastAPI(title='EduMine API', version='1.0')
rest.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*']
)

class StudentFeatures(BaseModel):
    total_clicks:        float = 0
    active_days:         float = 0
    mean_daily_clicks:   float = 0
    activity_diversity:  float = 0
    weeks_active:        float = 0
    click_std:           float = 0
    max_weekly:          float = 0
    min_weekly:          float = 0
    num_submissions:     float = 0
    avg_score:           float = 0
    min_score:           float = 0
    early_clicks:        float = 0
    gender_M:            int   = 0
    disability_Y:        int   = 0
    edu_level:           int   = 2
    num_of_prev_attempts:int   = 0
    studied_credits:     int   = 60

def run_prediction(features_dict):
    if model is None or not FEATURES:
        return None, "Model not loaded", []
    X = np.array([[features_dict.get(f, 0) for f in FEATURES]])
    prob     = float(model.predict_proba(X)[0, 1])
    risk_pct = round(prob * 100, 1)
    risk_lbl = 'High Risk' if prob > 0.6 else 'Moderate Risk' if prob > 0.4 else 'Low Risk'
    shap_vals = explainer.shap_values(X)[0]
    top_factors = sorted(zip(FEATURES, shap_vals.tolist()), key=lambda x: -abs(x[1]))[:5]
    return prob, risk_pct, risk_lbl, [{'feature': f, 'shap': round(s, 4)} for f, s in top_factors]

@rest.post('/predict')
def predict(s: StudentFeatures):
    prob, risk_pct, risk_lbl, top_factors = run_prediction(s.dict())
    return {
        'risk_probability': prob,
        'risk_percent':     risk_pct,
        'risk_label':       risk_lbl,
        'top_factors':      top_factors,
    }

@rest.get('/health')
def health():
    return {'status': 'ok', 'model': 'EduMine week3', 'loaded': model is not None}

# ─── Gradio UI (shown in the Space) ──────────────────────────────────────────
def gradio_predict(total_clicks, active_days, mean_daily_clicks, activity_diversity,
                   weeks_active, avg_score, num_submissions, early_clicks,
                   edu_level, studied_credits, num_of_prev_attempts):
    features = dict(
        total_clicks=total_clicks, active_days=active_days,
        mean_daily_clicks=mean_daily_clicks, activity_diversity=activity_diversity,
        weeks_active=weeks_active, click_std=5, max_weekly=0, min_weekly=0,
        num_submissions=num_submissions, avg_score=avg_score, min_score=avg_score,
        early_clicks=early_clicks, gender_M=0, disability_Y=0,
        edu_level=edu_level, num_of_prev_attempts=num_of_prev_attempts,
        studied_credits=studied_credits,
    )
    prob, risk_pct, risk_lbl, top_factors = run_prediction(features)
    shap_text = "\n".join([
        f"{'↑' if f['shap']>0 else '↓'} {f['feature']}: {f['shap']:+.4f}"
        for f in top_factors
    ])
    return f"**{risk_lbl}** — {risk_pct}% at-risk probability\n\n**Top SHAP factors:**\n{shap_text}"

with gr.Blocks(title="EduMine — At-Risk Student Predictor") as demo:
    gr.Markdown("## 📊 EduMine — At-Risk Student Early Warning System")
    gr.Markdown("Predict dropout risk using first 3 weeks of LMS engagement data.")
    with gr.Row():
        with gr.Column():
            total_clicks     = gr.Number(label="Total Clicks",          value=40)
            active_days      = gr.Number(label="Active Days",           value=10)
            mean_daily       = gr.Number(label="Avg Daily Clicks",      value=4)
            diversity        = gr.Number(label="Activity Types Used",   value=3)
            weeks_active     = gr.Number(label="Weeks Active",          value=3)
            avg_score        = gr.Number(label="Avg Assessment Score",  value=55)
            num_submissions  = gr.Number(label="Assessments Submitted", value=1)
            early_clicks     = gr.Number(label="Early Clicks (Wk 1-2)", value=15)
            edu_level        = gr.Slider(0, 4, step=1, label="Education Level (0-4)", value=2)
            credits          = gr.Number(label="Credits Enrolled",      value=60)
            prev_attempts    = gr.Number(label="Previous Attempts",     value=0)
            btn = gr.Button("Predict Risk →", variant="primary")
        with gr.Column():
            result = gr.Markdown("*Enter student data and click Predict.*")
    btn.click(gradio_predict,
              inputs=[total_clicks, active_days, mean_daily, diversity, weeks_active,
                      avg_score, num_submissions, early_clicks, edu_level, credits, prev_attempts],
              outputs=result)

# ─── Mount FastAPI on the Gradio app ─────────────────────────────────────────
# HuggingFace Gradio Spaces auto-detect the `app` variable and serve it on
# port 7860 via their own uvicorn instance — do NOT call uvicorn.run() here.
app = gr.mount_gradio_app(rest, demo, path="/")
