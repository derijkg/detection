from pathlib import Path
import pandas as pd

# Load preprocessed dataset
path = Path('/home/gderijck/detection/data_static/preprocessed/preprocessed_dataset.csv')
df = pd.read_csv(path)

# Fill NaNs for clean categorical grouping
df['scope'] = df['scope'].fillna('Human / None')
df['model_name'] = df['model_name'].fillna('Human / Original')

# ==============================================================================
# TABLE 1: CROSSTAB OF SPLIT vs SCOPE
# ==============================================================================
ct_scope = pd.crosstab(
    index=df['split'], 
    columns=df['scope'], 
    margins=True, 
    margins_name='Total'
)

print("=== TABLE 1: SPLIT vs SCOPE ===")
print(ct_scope)

latex_scope = ct_scope.to_latex(
    caption="Sample counts across dataset splits and paraphrase scopes.",
    label="tab:crosstab_split_scope",
    position="htbp"
)

with open("table_crosstab_split_scope.tex", "w") as f:
    f.write(latex_scope)
print("\n[Saved] table_crosstab_split_scope.tex")


# ==============================================================================
# TABLE 2: CROSSTAB OF SPLIT vs GENERATOR MODEL
# ==============================================================================
ct_model = pd.crosstab(
    index=df['split'], 
    columns=df['model_name'], 
    margins=True, 
    margins_name='Total'
)

print("\n=== TABLE 2: SPLIT vs GENERATOR MODEL ===")
print(ct_model)

latex_model = ct_model.to_latex(
    caption="Sample counts across dataset splits and generation models (including human baseline).",
    label="tab:crosstab_split_model",
    position="htbp"
)

with open("table_crosstab_split_model.tex", "w") as f:
    f.write(latex_model)
print("\n[Saved] table_crosstab_split_model.tex")