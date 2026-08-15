from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
path = Path('/home/gderijck/detection/data_static/raw/llm_added.parquet')
df = pd.read_parquet(path)

# --- HELPER FUNCTIONS ---
def get_word_count(series):
    return series.dropna().astype(str).apply(lambda x: len(x.split()) if x.strip() != '' else np.nan)

# Extract word counts for human baseline
df['human_word_count'] = get_word_count(df['abstract'])

# Define model columns dynamically
llm_cols = [c for c in df.columns if any(c.endswith(s) for s in ['_single', '_full', '_25', '_50', '_75'])]

# --- 1. HUMAN BASELINE OVERVIEW (TABLE 1) ---
num_samples = len(df)
num_sources = df['source'].nunique() if 'source' in df.columns else 'N/A'
year_min = int(df['year'].min()) if 'year' in df.columns else 'N/A'
year_max = int(df['year'].max()) if 'year' in df.columns else 'N/A'

human_mean_words = df['human_word_count'].mean()
human_std_words = df['human_word_count'].std()

print(f"Dataset Size: {num_samples} samples")
print(f"Sources: {num_sources} venues ({year_min}-{year_max})")
print(f"Human Abstract Mean Words: {human_mean_words:.2f} ± {human_std_words:.2f}")

# --- 2. GENERATE MODEL PARAPHRASE METRICS (TABLE 2) ---
records = []
records = []
for col in llm_cols:
    # Parse Model and Regime
    model_name, regime = col.split('_')
    
    # Check non-null and non-empty generations
    valid_mask = df[col].notna() & (df[col].astype(str).str.strip() != '') & (df[col].astype(str) != 'None')
    valid_count = valid_mask.sum()
    completion_rate = (valid_count / len(df)) * 100
    
    records.append({
        'Model': model_name,
        'Regime': regime,
        'Total Count': valid_count,
        'Completion': completion_rate
    })

stats_df = pd.DataFrame(records)

# Format table for LaTeX
latex_table_models = stats_df.to_latex(
    index=False,
    float_format="%.2f",
    caption="Summary of total generated paraphrases and completion rates across models and regimes.",
    label="tab:llm_paraphrase_summary",
    column_format="llrr",
    position="htbp"
)

with open("table_llm_summary.tex", "w") as f:
    f.write(latex_table_models)
print("\n[Saved] table_llm_summary.tex")

# --- 3. SOURCE DISTRIBUTION TABLE ---
if 'source' in df.columns:
    source_df = df['source'].value_counts().reset_index()
    source_df.columns = ['Source Venue', 'Abstract Count']
    source_df['Share (%)'] = (source_df['Abstract Count'] / len(df) * 100).round(2)
    
    latex_sources = source_df.head(10).to_latex(
        index=False,
        float_format="%.2f",
        caption="Top source venues represented in the human dataset.",
        label="tab:data_sources",
        position="htbp"
    )
    with open("table_sources.tex", "w") as f:
        f.write(latex_sources)
    print("[Saved] table_sources.tex")

# --- 4. VISUALIZATION (LENGTH DISTRIBUTION) ---
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(6.5, 3.5))

# Plot Human baseline vs Full Paraphrase Regimes
sns.kdeplot(df['human_word_count'], label='Human Abstract', linewidth=2, color='black', ax=ax)

full_cols = [c for c in llm_cols if c.endswith('_full')]
for col in full_cols:
    model_label = col.replace('_full', '')
    sns.kdeplot(get_word_count(df[col]), label=f'Full: {model_label}', linestyle='--', ax=ax)

ax.set_title('Word Count Distribution: Human vs. Full LLM Paraphrases', fontsize=11)
ax.set_xlabel('Word Count', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()

# Export vector graphic PDF for LaTeX
plt.savefig('fig_word_length_dist.pdf', format='pdf', bbox_inches='tight')
print("[Saved] fig_word_length_dist.pdf")