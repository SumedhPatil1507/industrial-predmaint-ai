"""
Quick CLI training script.
Run: python scripts/train_from_csv.py data/synthetic_industrial_machine_data.csv
"""
import sys
import pandas as pd

sys.path.insert(0, ".")
from backend.ml_engine import train_model

if len(sys.argv) < 2:
    print("Usage: python scripts/train_from_csv.py <path_to_csv>")
    sys.exit(1)

path = sys.argv[1]
print(f"Loading {path}...")
df = pd.read_csv(path)
print(f"Shape: {df.shape}")

metrics = train_model(df)
print("\n✅ Model trained!")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")
