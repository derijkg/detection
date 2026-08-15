#!/usr/bin/env python3
# scripts/svmfeatures.py

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 0. Define Custom Transformers for Joblib Unpickling
# ==========================================
class TextExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, key='text'):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

class StylometricExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, granularity='full'):
        self.granularity = granularity
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

class StylometricScaler(BaseEstimator, TransformerMixin):
    def __init__(self, weight=1.0):
        self.weight = weight
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

# Register classes in __main__ namespace so unpickler finds them
import __main__
setattr(__main__, 'TextExtractor', TextExtractor)
setattr(__main__, 'StylometricExtractor', StylometricExtractor)
setattr(__main__, 'StylometricScaler', StylometricScaler)

# ==========================================
# 1. Setup Paths & Load Objects
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "svm" / "full" / "model.joblib"
PIPELINE_PATH = PROJECT_ROOT / "data_static" / "model_features" / "svm_full" / "pipeline.joblib"

print(f"Loading calibrated SVM model from : {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print(f"Loading feature pipeline from     : {PIPELINE_PATH}")
pipeline = joblib.load(PIPELINE_PATH)

# ==========================================
# 2. Extract Coefficients from CalibratedClassifierCV
# ==========================================
if hasattr(model, 'calibrated_classifiers_'):
    fold_weights = [clf.estimator.coef_[0] for clf in model.calibrated_classifiers_]
    weights = np.mean(fold_weights, axis=0)
    print(f"Successfully extracted and averaged weights across {len(fold_weights)} calibration fold(s).")
elif hasattr(model, 'coef_'):
    weights = model.coef_[0]
else:
    raise ValueError("Could not retrieve coefficients (coef_) from the model.")

# ==========================================
# 3. Extract Feature Names from Pipeline
# ==========================================
STYLOMETRIC_NAMES = [
    "log_mean_sent_len",
    "log_var_sent_len",
    "burstiness",
    "mean_word_len",
    "log_var_word_len",
    "ttr",
    "hapax_ratio",
    "transition_ratio",
    "space_ratio",
    "double_space_ratio",
    "punc_ratio"
]

feature_names = []

if hasattr(pipeline, 'named_steps') and 'union' in pipeline.named_steps:
    union = pipeline.named_steps['union']
    for name, trans in union.transformer_list:
        if 'word' in name:
            tfidf = trans.named_steps['tfidf']
            feature_names.extend([f"word: {w}" for w in tfidf.get_feature_names_out()])
        elif 'char' in name:
            tfidf = trans.named_steps['tfidf']
            feature_names.extend([f"char: '{c}'" for c in tfidf.get_feature_names_out()])
        elif 'sty' in name:
            feature_names.extend([f"sty: {s}" for s in STYLOMETRIC_NAMES])
else:
    print("[WARNING] Could not parse FeatureUnion steps directly. Using index fallback.")
    feature_names = [f"feat_{i}" for i in range(len(weights))]

if len(feature_names) != len(weights):
    print(f"[WARNING] Feature names count ({len(feature_names)}) != weights count ({len(weights)}).")
    feature_names = [f"feat_{i}" for i in range(len(weights))]

# ==========================================
# 4. Build DataFrame & Sort
# ==========================================
df_feat = pd.DataFrame({
    'feature': feature_names,
    'weight': weights,
    'abs_weight': np.abs(weights)
}).sort_values(by='weight', ascending=False)

top_llm = df_feat.head(15)
top_human = df_feat.tail(15).iloc[::-1]

print("\n" + "="*65)
print(" TOP 15 LLM INDICATORS (Positive SVM Weights)")
print("="*65)
for idx, row in top_llm.iterrows():
    print(f"  {row['feature']:<40} : +{row['weight']:.4f}")

print("\n" + "="*65)
print(" TOP 15 HUMAN INDICATORS (Negative SVM Weights)")
print("="*65)
for idx, row in top_human.iterrows():
    print(f"  {row['feature']:<40} : {row['weight']:.4f}")

# ==========================================
# 5. Stylometric Features Ranking
# ==========================================
df_sty = df_feat[df_feat['feature'].str.startswith('sty:')].copy()
df_sty['feature'] = df_sty['feature'].str.replace('sty: ', '')
df_sty = df_sty.sort_values(by='weight', ascending=False)

print("\n" + "="*65)
print(" STYLOMETRIC FEATURES RANKING")
print("="*65)
for idx, row in df_sty.iterrows():
    direction = "LLM ->" if row['weight'] > 0 else "Human ->"
    print(f"  {row['feature']:<30} : {row['weight']:+8.4f}  ({direction})")

# ==========================================
# 6. Save Plots
# ==========================================
top_plot = pd.concat([top_llm.head(10), top_human.head(10).iloc[::-1]])

plt.figure(figsize=(10, 6), dpi=300)
colors = ['#2b5c8f' if w > 0 else '#d95f02' for w in top_plot['weight']]
plt.barh(top_plot['feature'], top_plot['weight'], color=colors, edgecolor='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel("SVM Feature Weight (Negative = Human, Positive = LLM)", fontsize=11)
plt.title("Top Linear SVM Feature Weights (Full Abstract Scope)", fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()

plot_path = PROJECT_ROOT / "outputs" / "svm" / "full" / "svm_top_features.png"
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"\n[SAVED PLOT] Top features plot saved to: '{plot_path}'")

plt.figure(figsize=(8, 5), dpi=300)
colors_sty = ['#2b5c8f' if w > 0 else '#d95f02' for w in df_sty['weight']]
plt.barh(df_sty['feature'], df_sty['weight'], color=colors_sty, edgecolor='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel("SVM Feature Weight", fontsize=11)
plt.title("Stylometric Feature Weights (Full Scope)", fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()

sty_plot_path = PROJECT_ROOT / "outputs" / "svm" / "full" / "svm_stylometrics.png"
plt.savefig(sty_plot_path, dpi=300)
plt.close()
print(f"[SAVED PLOT] Stylometric weights plot saved to: '{sty_plot_path}'")

# ==========================================
# 7. Generate LaTeX Table
# ==========================================
latex_table = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Top Linear SVM feature weights indicating LLM-generated vs. Human Dutch abstracts.}",
    r"\label{tab:svm_feature_weights}",
    r"\begin{tabular}{lr|lr}",
    r"\toprule",
    r"\multicolumn{2}{c|}{\textbf{Top LLM Predictors (+) }} & \multicolumn{2}{c}{\textbf{Top Human Predictors (--)}} \\",
    r"\textbf{Feature} & \textbf{Weight} & \textbf{Feature} & \textbf{Weight} \\",
    r"\midrule"
]

for (i, row_l), (j, row_h) in zip(top_llm.head(10).iterrows(), top_human.head(10).iterrows()):
    f_llm = str(row_l['feature']).replace('_', r'\_')
    f_hum = str(row_h['feature']).replace('_', r'\_')
    latex_table.append(f"{f_llm} & +{row_l['weight']:.4f} & {f_hum} & {row_h['weight']:.4f} \\\\")

latex_table.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}"
])

print("\n--- LATEX TABLE OUTPUT ---\n")
print("\n".join(latex_table))