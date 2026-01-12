from __future__ import annotations
from pathlib import Path
import joblib
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from src.data_load import load_application_train
from src.features import train_val_split
from src.thresholding import find_best_threshold

def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "application_train.csv"
    model_path = root / "reports" / "xgb_model.joblib"

    df = load_application_train(data_path)
    X_train, X_val, y_train, y_val = train_val_split(df)

    pipe = joblib.load(model_path)
    y_prob = pipe.predict_proba(X_val)[:, 1]

    best_t, best_cost = find_best_threshold(y_val.values, y_prob, cost_fn=5.0, cost_fp=1.0)

    y_pred = (y_prob >= best_t).astype(int)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    print(f"Best threshold (FN=5, FP=1): {best_t:.3f}")
    print(f"Expected cost (units):        {best_cost:.1f}")
    print(f"Precision: {prec:.3f} | Recall: {rec:.3f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

if __name__ == "__main__":
    main()
