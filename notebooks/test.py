from pathlib import Path
import pandas as pd

df = pd.read_parquet(Path(r'E:\code\dta\detection\data_static\preprocessed\preprocessed_dataset.parquet'))

print(pd.crosstab(df['split'], df['scope']))