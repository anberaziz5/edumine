"""
Extract temporal features from OULAD clickstream.
All features computed for a given prediction week W.
Models trained with W=3 (early warning) and W=all (full semester).
"""
import pandas as pd, numpy as np

def extract_features(info, vle, assessments, assdtl, cutoff_week=None):
    """
    cutoff_week: int or None
        If int, only use VLE data up to (cutoff_week * 7) days
        If None, use all available data
    Returns: feature DataFrame indexed by (id_student, code_module, code_presentation)
    """
    vle_cut = vle.copy()
    if cutoff_week is not None:
        vle_cut = vle_cut[vle_cut['date'] <= cutoff_week * 7]
        
    features = {}
    
    # ─── Feature group 1: Volume ──────────────────────────────────────
    vol = vle_cut.groupby(['id_student','code_module','code_presentation']).agg(
        total_clicks   = ('sum_click','sum'),
        active_days    = ('date','nunique'),
        mean_daily_clicks = ('sum_click','mean'),
    ).reset_index()

    # ─── Feature group 2: Diversity (activity types used) ────────────
    div = vle_cut.groupby(['id_student','code_module','code_presentation'])['activity_type'].nunique().reset_index()
    div.columns = ['id_student','code_module','code_presentation','activity_diversity']

    # ─── Feature group 3: Temporal regularity ────────────────────────
    vle_cut['week'] = vle_cut['date'] // 7 + 1
    weekly = vle_cut.groupby(['id_student','code_module','code_presentation','week'])['sum_click'].sum().reset_index()
    reg = weekly.groupby(['id_student','code_module','code_presentation']).agg(
        weeks_active   = ('week','nunique'),
        click_std      = ('sum_click','std'),   # high std = irregular
        max_weekly     = ('sum_click','max'),
        min_weekly     = ('sum_click','min'),
    ).reset_index()
    reg['click_std'] = reg['click_std'].fillna(0)

    # ─── Feature group 4: Assessment behavior ────────────────────────
    merged_assess = assessments.merge(assdtl[['id_assessment','date']], on='id_assessment')
    if cutoff_week is not None:
        merged_assess = merged_assess[merged_assess['date'] <= cutoff_week * 7]
        
    asmfeat = merged_assess.groupby(['id_student','code_module','code_presentation']).agg(
        num_submissions = ('id_assessment','count'),
        avg_score       = ('score','mean'),
        min_score       = ('score','min'),
    ).reset_index()

    # ─── Feature group 5: Early engagement (first 2 weeks) ───────────
    early = vle_cut[vle_cut['week'] <= 2].groupby(
        ['id_student','code_module','code_presentation'])['sum_click'].sum().reset_index()
    early.columns = ['id_student','code_module','code_presentation','early_clicks']

    # ─── Merge all feature groups ─────────────────────────────────────
    df = info[['id_student','code_module','code_presentation','at_risk',
               'gender','highest_education','age_band','num_of_prev_attempts',
               'studied_credits','disability']].copy()
               
    for feat_df in [vol, div, reg, asmfeat, early]:
        df = df.merge(feat_df, on=['id_student','code_module','code_presentation'], how='left')

    # ─── Encode categoricals ──────────────────────────────────────────
    df['gender_M']     = (df['gender']=='M').astype(int)
    df['disability_Y'] = (df['disability']=='Y').astype(int)
    
    edu_map = {'No Formal quals':0,'Lower Than A Level':1,'A Level or Equivalent':2,
               'HE Qualification':3,'Post Graduate Qualification':4}
    df['edu_level'] = df['highest_education'].map(edu_map).fillna(0)
    
    df = df.fillna(0)
    return df
