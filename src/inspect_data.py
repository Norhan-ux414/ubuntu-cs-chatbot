import os
import pandas as pd

FILENAME = "dialogueText.csv"   
PATH = os.path.join("data", FILENAME)

df = pd.read_csv(PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nSample rows:")
print(df.head(3).to_string(index=False))
