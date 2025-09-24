import pandas as pd
from pycaret.regression import setup, compare_models, predict_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === 1) Carregar o dataset ===
# Se for CSV:
rio_iqr = pd.read_csv("rio_iqr_3.csv")
# Se for Parquet:
# rio_iqr = pd.read_parquet("rio_iqr.parquet")

print("Shape:", rio_iqr.shape)
print(rio_iqr.head())

# === 2) Definir target e features ===
target_col = "price"
leaky_cols = ["price_per_person", "price_per_bedroom", "price_per_bathroom"]

# Mantém todas as colunas numéricas exceto target e vazamento
feature_cols = [c for c in rio_iqr.columns if c not in [target_col] + leaky_cols]

# === 3) Setup do PyCaret ===
s = setup(
    data=rio_iqr,
    target=target_col,
    session_id=42,
    preprocess=True,
    normalize=True,            # z-score
    normalize_method="zscore",
    imputation_type="simple",  # imputação simples
    numeric_imputation="median",
    verbose=False
)

# === 4) Comparar modelos ===
best_model = compare_models(sort="RMSE", n_select=1)

# === 5) Previsão in-sample (só pra validar rapidamente) ===
preds = predict_model(best_model, data=rio_iqr)

rmse = mean_squared_error(rio_iqr[target_col], preds["prediction_label"], squared=False)
mae  = mean_absolute_error(rio_iqr[target_col], preds["prediction_label"])
r2   = r2_score(rio_iqr[target_col], preds["prediction_label"])

print(f"\nMelhor modelo: {type(best_model).__name__}")
print(f"RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")
