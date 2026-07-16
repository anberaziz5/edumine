from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json, numpy as np
from huggingface_hub import hf_hub_download

app = FastAPI(title='EduMine API', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# Download artifacts from HuggingFace on startup
REPO = 'NareenAsad/edumine-xgboost'
try:
    model    = joblib.load(hf_hub_download(REPO, 'week3_model.pkl'))
    explainer= joblib.load(hf_hub_download(REPO, 'week3_explainer.pkl'))
    with open(hf_hub_download(REPO, 'week3_features.json')) as f:
        FEATURES = json.load(f)
except Exception as e:
    print(f"Warning: Could not load models on startup. Make sure you have uploaded them to {REPO}")
    print(e)
    model, explainer, FEATURES = None, None, []

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

@app.post('/predict')
def predict(s: StudentFeatures):
    if not model:
        return {"error": "Model not loaded"}
    X = np.array([[getattr(s, f) for f in FEATURES]])
    prob     = float(model.predict_proba(X)[0,1])
    risk_pct = round(prob*100, 1)
    risk_lbl = 'High Risk' if prob>0.6 else 'Moderate Risk' if prob>0.4 else 'Low Risk'
    
    # Per-student SHAP
    shap_vals = explainer.shap_values(X)[0]
    top_factors = sorted(zip(FEATURES, shap_vals.tolist()), key=lambda x:-abs(x[1]))[:5]
    
    return {
        'risk_probability': prob,
        'risk_percent':     risk_pct,
        'risk_label':       risk_lbl,
        'top_factors': [{'feature':f,'shap':round(s,4)} for f,s in top_factors]
    }

@app.get('/health')
def health(): return {'status':'ok','model':'EduMine week3'}
