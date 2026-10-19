"""Step 0: create the raw dataset in data/01_raw/.

Dataset: Diabetes data (Efron, Hastie, Johnstone & Tibshirani, 2004, "Least Angle
Regression", Annals of Statistics). Original source:
https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
A copy ships with scikit-learn (BSD-3-Clause), so no internet access is needed.

Usage (from the project root):
    python scripts/download_data.py
"""

from pathlib import Path

from sklearn.datasets import load_diabetes

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "01_raw"

# readable names for the anonymised blood-serum columns s1..s6
RENAME = {
    "s1": "total_cholesterol",
    "s2": "ldl",
    "s3": "hdl",
    "s4": "chol_hdl_ratio",
    "s5": "log_triglycerides",
    "s6": "glucose",
    "bp": "blood_pressure",
    "target": "progression",
}


def main() -> None:
    df = load_diabetes(as_frame=True, scaled=False).frame.rename(columns=RENAME)
    df["sex"] = df["sex"].astype(int)
    df.insert(0, "patient_id", range(1, len(df) + 1))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "diabetes_patients.csv", index=False)
    df.sample(n=60, random_state=0).to_csv(RAW_DIR / "diabetes_sample.csv", index=False)
    print(f"Saved {len(df)} rows to {RAW_DIR / 'diabetes_patients.csv'}")


if __name__ == "__main__":
    main()
