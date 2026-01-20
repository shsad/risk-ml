from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.data_load import load_application_train 
from src.features import train_val_split

def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "application_train.csv"
    model_path = root / "reports" / "xgb_model.joblib"
    out_csv = root / "reports" / "xgb_feature_importance.csv"

    df = load_application_train(str(data_path)) # load training dataset
    X_train, X_val, y_train, y_val = train_val_split(df) # split it into train and validation sets

    pipe = joblib.load(model_path) # loads the trained pipeline

    # Get feature names from preprocessing (numeric, categorical one-hot encoded)
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    # numeric names
    num_names = pre.transformers_[0][2]
    # categorical onehot names
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_names = pre.transformers_[1][2]
    ohe_names = ohe.get_feature_names_out(cat_names).tolist()

    feature_names = list(num_names) + list(ohe_names)

    # Retrieve feature importances from the XGBoost model, combine names and importances into a DataFrame
    importances = model.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})

    # Sort by importance
    df_imp = df_imp.sort_values("importance", ascending=False)

    # Write results to reports/xgb_feature_importance.csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(out_csv, index=False)

    print("Top 20 features:")
    print(df_imp.head(20).to_string(index=False))
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
