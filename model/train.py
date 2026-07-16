"""Train XGBoost at-risk classifier with SHAP explainability."""
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import shap, joblib, json

FEATURE_COLS = [
    'total_clicks','active_days','mean_daily_clicks','activity_diversity',
    'weeks_active','click_std','max_weekly','min_weekly',
    'num_submissions','avg_score','min_score','early_clicks',
    'gender_M','disability_Y','edu_level','num_of_prev_attempts','studied_credits'
]

def train_model(csv_path, model_name):
    df = pd.read_csv(csv_path)
    X  = df[FEATURE_COLS].fillna(0)
    y  = df['at_risk']

    # ─── Class imbalance: weight at-risk class higher ─────────────
    pos = y.sum(); neg = len(y) - pos
    scale = neg / pos
    print(f'At-risk: {pos}, Not: {neg}, scale_pos_weight: {scale:.2f}')

    # ─── 5-fold cross validation ──────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1, cv_auc = [], []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale, use_label_encoder=False,
            eval_metric='logloss', random_state=42
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=30, verbose=False)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:,1]
        
        f1  = f1_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)
        
        print(f'  Fold {fold+1}: F1={f1:.4f}  AUC={auc:.4f}')
        cv_f1.append(f1); cv_auc.append(auc)

    print(f'CV F1: {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}')
    print(f'CV AUC: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}')

    # ─── Train final model on full data ───────────────────────────
    final_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale, use_label_encoder=False,
        eval_metric='logloss', random_state=42
    )
    final_model.fit(X, y)

    # ─── SHAP values ──────────────────────────────────────────────
    explainer   = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)
    
    mean_shap   = pd.Series(np.abs(shap_values).mean(0), index=FEATURE_COLS)
    print('\nTop 5 SHAP features:')
    print(mean_shap.sort_values(ascending=False).head())

    # ─── Save artifacts ───────────────────────────────────────────
    joblib.dump(final_model,  f'/kaggle/working/{model_name}_model.pkl')
    joblib.dump(explainer,    f'/kaggle/working/{model_name}_explainer.pkl')
    with open(f'/kaggle/working/{model_name}_features.json','w') as f:
        json.dump(FEATURE_COLS, f)
        
    print(f'Saved {model_name} artifacts')
    return final_model, explainer, cv_f1, cv_auc

if __name__ == '__main__':
    # Train both models
    model_full,  exp_full,  f1_full,  auc_full  = train_model('/kaggle/working/features_full.csv',  'full')
    model_week3, exp_week3, f1_week3, auc_week3 = train_model('/kaggle/working/features_week3.csv', 'week3')
