from __future__ import annotations
import pandas as pd

# Create reusable data loader

def load_application_train(path: str) -> pd.DataFrame:
    """
    Load the Home Credit application_train.csv dataset
    Perform minimal cleaning:
      - standardize column names to lowercase
      - ensure TARGET is integer type
    """
    df = pd.read_csv(path)

    # Consistent column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure target exists and is integer
    if "target" not in df.columns:
        raise ValueError("TARGET column missing in dataset")
    df["target"] = df["target"].astype(int)

    return df

if __name__ == "__main__":
    # Example usage
    data = load_application_train("../data/raw/application_train.csv")
    print(data.shape)
    print(data["target"].value_counts(normalize=True))




