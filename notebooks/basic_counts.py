import pandas as pd

# 1. Load dataset
file_path = "/home/gderijck/detection/data_static/preprocessed/preprocessed_dataset.csv"
df = pd.read_csv(file_path)

# Ensure numeric / string types
df['_id'] = df['_id'].astype(str)
df['split'] = df['split'].astype(str)
df['model_name'] = df['model_name'].astype(str)
df['scope'] = df['scope'].astype(str)

# Map 'single' to 'sentence' if present for clean labeling
df['scope_label'] = df['scope'].replace({'single': 'Sentence', 'full': 'Full Abstract'})

# ==============================================================================
# TABLE 1: Unique Abstract Counts per Split
# ==============================================================================
unique_abstracts = df.groupby('split')['_id'].nunique().reset_index()
unique_abstracts.columns = ['Split', 'Unique Abstracts']

# Add Total Row
total_unique = df['_id'].nunique()
total_row = pd.DataFrame([{'Split': 'Total', 'Unique Abstracts': total_unique}])
unique_abstracts_df = pd.concat([unique_abstracts, total_row], ignore_index=True)

# Format numbers with commas
unique_abstracts_df['Unique Abstracts ($N$)'] = unique_abstracts_df['Unique Abstracts'].apply(lambda x: f"{x:,}")

# Generate LaTeX for Table 1
latex_table_1 = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Number of unique abstracts per dataset split.}}
\\label{{tab:unique_abstracts_per_split}}
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Split}} & \\textbf{{Unique Abstracts ($N$)}} \\\\
\\midrule
"""
for _, row in unique_abstracts_df.iterrows():
    if row['Split'] == 'Total':
        latex_table_1 += "\\midrule\n"
        latex_table_1 += f"\\textbf{{{row['Split']}}} & \\textbf{{{row['Unique Abstracts ($N$)']}}} \\\\\n"
    else:
        latex_table_1 += f"{row['Split']} & {row['Unique Abstracts ($N$)']} \\\\\n"

latex_table_1 += """\\bottomrule
\\end{tabular}
\\end{table}
"""

print("==================== TABLE 1: UNIQUE ABSTRACTS PER SPLIT ====================")
print(latex_table_1)


# ==============================================================================
# TABLE 2: Sample Counts per Model, Scope, and Split
# ==============================================================================
# Group by model_name, scope_label, and split
counts = df.groupby(['model_name', 'scope_label', 'split']).size().unstack(fill_value=0)

# Calculate total per model and scope across splits
counts['Total'] = counts.sum(axis=1)

# Reset index to make model_name and scope_label columns
counts_flat = counts.reset_index()

# Reorder columns dynamically (splits + Total)
splits_found = [c for c in ['train', 'val', 'validation', 'dev', 'test'] if c in counts.columns]
other_cols = [c for c in counts.columns if c not in splits_found + ['Total']]
ordered_cols = ['model_name', 'scope_label'] + splits_found + other_cols + ['Total']
counts_flat = counts_flat[ordered_cols]

# Build LaTeX for Table 2
col_headers = ["Model", "Scope"] + [c.capitalize() for c in splits_found + other_cols] + ["Total"]
col_format = "ll" + "r" * (len(col_headers) - 2)

latex_table_2 = f"""\\begin{{table}}[htbp]
\\centering
\\small
\\caption{{Sample counts per generator model, split, and scope (Full Abstract vs. Sentence).}}
\\label{{tab:sample_counts_per_model}}
\\begin{{tabular}}{{{col_format}}}
\\toprule
"""
latex_table_2 += " & ".join([f"\\textbf{{{h}}}" for h in col_headers]) + " \\\\\n\\midrule\n"

current_model = None
for _, row in counts_flat.iterrows():
    model_disp = row['model_name'] if row['model_name'] != current_model else ""
    current_model = row['model_name']
    
    scope_disp = row['scope_label']
    vals = [f"{row[c]:,}" for c in splits_found + other_cols + ['Total']]
    
    latex_table_2 += f"{model_disp} & {scope_disp} & " + " & ".join(vals) + " \\\\\n"

# Add overall totals row at the bottom
overall_totals = df.groupby(['scope_label', 'split']).size().unstack(fill_value=0)
overall_totals['Total'] = overall_totals.sum(axis=1)

latex_table_2 += "\\midrule\n"
for scope_val in overall_totals.index:
    vals = [f"{overall_totals.loc[scope_val, c]:,}" if c in overall_totals.columns else "0" for c in splits_found + other_cols + ['Total']]
    latex_table_2 += f"\\textbf{{Total ({scope_val})}} & & " + " & ".join(vals) + " \\\\\n"

latex_table_2 += """\\bottomrule
\\end{tabular}
\\end{table}
"""

print("\n==================== TABLE 2: SAMPLE COUNTS PER MODEL & SCOPE ====================")
print(latex_table_2)