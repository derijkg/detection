from pathlib import Path
import pandas as pd
import numpy as np
import random

# 1. Load data
path = Path('/home/gderijck/detection/data_static/raw/llm_added.parquet')
df = pd.read_parquet(path)

# Max character length for "not too long" abstract constraint
MAX_ABSTRACT_CHARS = 350  

# List of strings that signal invalid/failed generations
INVALID_STRINGS = {
    'NONE', 
    'FAILED_GENERATION', 
    'GENERATION_FAILED', 
    'FAILED_VALIDATION', 
    'VALIDATION_FAILED'
}

# Helper function to check if a single scalar value is valid
def is_valid_scalar(val):
    if val is None or pd.isna(val):
        return False
    if isinstance(val, str):
        val_clean = val.strip().upper()
        if val_clean in INVALID_STRINGS:
            return False
    return True

# Helper function to validate numpy array and all its items
def is_valid_array(arr):
    if arr is None or not isinstance(arr, (np.ndarray, list)):
        return False
    if len(arr) == 0:
        return False
    return all(is_valid_scalar(item) for item in arr)

# Helper function to escape LaTeX special characters
def escape_latex(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace('\\', r'\textbackslash{}')
    for char in ['&', '%', '$', '#', '_', '{', '}']:
        text = text.replace(char, f'\\{char}')
    text = text.replace('~', r'\textasciitilde{}')
    text = text.replace('^', r'\textasciicircum{}')
    return text

# 2. Extract model names ({model}_single and {model}_full)
single_cols = [col for col in df.columns if col.endswith('_single')]
models = [col[:-7] for col in single_cols if f"{col[:-7]}_full" in df.columns]

# 3. Collect valid candidates satisfying length and failure checks
candidates = []

for model in models:
    single_col = f"{model}_single"
    full_col = f"{model}_full"
    
    for idx, row in df.iterrows():
        s_val = row[single_col]
        f_val = row[full_col]
        orig_val = row.get('abstract', None)
        
        if is_valid_array(s_val) and is_valid_scalar(f_val) and is_valid_scalar(orig_val):
            abstract_str = str(f_val).strip()
            orig_abstract_str = str(orig_val).strip()
            
            # Ensure both abstracts are concise
            if len(abstract_str) <= MAX_ABSTRACT_CHARS and len(orig_abstract_str) <= MAX_ABSTRACT_CHARS:
                joined_sentence = " ".join(str(item) for item in s_val)
                candidates.append({
                    'model': model,
                    'sentence': joined_sentence,
                    'full_abstract': abstract_str,
                    'original_abstract': orig_abstract_str
                })

# Fallback: If no candidate was under MAX_ABSTRACT_CHARS, pick from valid entries with shortest abstracts
if not candidates:
    for model in models:
        single_col, full_col = f"{model}_single", f"{model}_full"
        for idx, row in df.iterrows():
            s_val, f_val = row[single_col], row[full_col]
            orig_val = row.get('abstract', None)
            if is_valid_array(s_val) and is_valid_scalar(f_val) and is_valid_scalar(orig_val):
                candidates.append({
                    'model': model,
                    'sentence': " ".join(str(item) for item in s_val),
                    'full_abstract': str(f_val).strip(),
                    'original_abstract': str(orig_val).strip()
                })
    candidates.sort(key=lambda x: len(x['full_abstract']) + len(x['original_abstract']))
    candidates = candidates[:10]

# 4. Pick exactly ONE random combination
selected = random.choice(candidates)

# Escaped values for LaTeX insertion
model_esc = escape_latex(selected['model'])
orig_esc = escape_latex(selected['original_abstract'])
sentence_esc = escape_latex(selected['sentence'])
full_esc = escape_latex(selected['full_abstract'])

# 5. Build LaTeX table matching the updated format
latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\begin{{tabular}}{{|p{{4.8cm}}|p{{4.8cm}}|p{{4.8cm}}|}}
\\hline
\\textbf{{Original Abstract}} & \\textbf{{Sentence ({model_esc})}} & \\textbf{{Full Abstract ({model_esc})}} \\\\ \\hline
{orig_esc} & {sentence_esc} & {full_esc} \\\\ \\hline
\\end{{tabular}}
\\caption{{Sample sentence, generated abstract, and original abstract for model {model_esc}.}}
\\label{{tab:single_model_sample}}
\\end{{table}}"""

print(latex_table)