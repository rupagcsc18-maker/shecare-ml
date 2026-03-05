"""
Run this script once to train both models from your collected datasets.
"""
from dataset_integrator import train_all_models

pcos_trainer, cycle_trainer = train_all_models(
    pcos_csv_path  = "data/PCOS_infertility.csv",
    pcos_xlsx_path = "data/PCOS_data_without_infertility.xlsx",
    cycle_csv_path = "data/FedCycleData071012.csv",
    save_dir       = "models/",
)



## Updated Folder Structure
# ```
# menstruation_tracker/
# ├── data/
# │   ├── PCOS_infertility.csv                ← your dataset 1a
# │   ├── PCOS_data_without_infertility.xlsx  ← your dataset 1b
# │   └── FedCycleData071012.csv              ← your dataset 2
# ├── models/
# │   ├── pcos_classifier.pkl                 ← trained & saved
# │   └── cycle_length_xgb.pkl               ← trained & saved
# ├── dataset_integrator.py                  ← NEW
# ├── train_from_datasets.py                 ← NEW (run once)
# ├── irregular_detector.py
# ├── condition_classifier.py
# ├── adaptive_predictor.py
# └── tests/