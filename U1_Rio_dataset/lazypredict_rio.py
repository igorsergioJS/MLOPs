import os
import sys
import numpy as np
import pandas as pd

# Tentar importar LazyRegressor com mensagem amigável se faltar a lib
try:
	from lazypredict.Supervised import LazyRegressor
except ImportError:
	print(
		"Dependência ausente: lazypredict.\n"
		"Instale com: pip install lazypredict\n"
		"Obs.: Requer scikit-learn."
	)
	raise

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
	# === 1) Caminhos ===
	here = os.path.dirname(os.path.abspath(__file__))
	csv_path = os.path.join(here, "rio_iqr_3.csv")
	out_ranking = os.path.join(here, "lazypredict_val_ranking.csv")
	out_fallback = os.path.join(here, "lazypredict_fallback.csv")

	if not os.path.exists(csv_path):
		print(f"CSV não encontrado em: {csv_path}")
		sys.exit(1)

	# === 2) Carregar o dataset ===
	df = pd.read_csv(csv_path)
	print("Shape:", df.shape)

	# === 3) Definir target e colunas com vazamento (iguais ao PyCaret) ===
	target_col = "price"
	leaky_cols = ["price_per_person", "price_per_bedroom", "price_per_bathroom"]

	if target_col not in df.columns:
		print(f"Coluna alvo '{target_col}' não está no CSV.")
		sys.exit(1)

	# === 4) Selecionar features numéricas, removendo target e vazamentos ===
	base_features = [c for c in df.columns if c not in [target_col] + leaky_cols]
	X_all = df[base_features].select_dtypes(include=[np.number]).copy()
	y_all = df[target_col].values

	if X_all.shape[1] == 0:
		print("Não há features numéricas após remover target e colunas com vazamento.")
		sys.exit(1)

	print(f"Features numéricas: {X_all.shape[1]} | Linhas: {X_all.shape[0]}")

	# === 5) Split train/val ===
	X_train, X_val, y_train, y_val = train_test_split(
		X_all, y_all, test_size=0.2, random_state=42
	)

	# === 6) Pré-processamento: imputação mediana + padronização (z-score) ===
	preprocess = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)

	X_train_proc = preprocess.fit_transform(X_train)
	X_val_proc = preprocess.transform(X_val)

	# === 7) LazyPredict ===
	lazy = LazyRegressor(verbose=0, ignore_warnings=True, predictions=True, random_state=42)
	models, preds = lazy.fit(X_train_proc, X_val_proc, y_train, y_val)

	# === 8) Métricas por modelo (com tolerância a falhas) ===
	rows = []
	bad = []
	for name, y_pred in preds.items():
		try:
			if y_pred is None:
				bad.append((name, "None prediction"))
				continue
			y_pred = np.asarray(y_pred).ravel()
			if y_pred.shape[0] != y_val.shape[0]:
				bad.append((name, f"shape {y_pred.shape} != {y_val.shape}"))
				continue
			rmse = mean_squared_error(y_val, y_pred, squared=False)
			mae = mean_absolute_error(y_val, y_pred)
			r2 = r2_score(y_val, y_pred)
			rows.append({"model": name, "RMSE_val": rmse, "MAE_val": mae, "R2_val": r2})
		except Exception as e:
			bad.append((name, str(e)))

	# === 9) Ranking ou fallback ===
	if rows:
		lazy_table = pd.DataFrame(rows).sort_values("RMSE_val").reset_index(drop=True)
		print("Top 15 (por RMSE_val):")
		print(lazy_table.head(15).to_string(index=False))
		lazy_table.to_csv(out_ranking, index=False)
		print(f"Ranking salvo em {out_ranking}")
	else:
		print("Nenhuma predição válida em 'preds'. Mostrando fallback com a tabela do LazyPredict:")
		cols_lower = [c.lower() for c in models.columns]
		if "rmse" in cols_lower:
			rmse_col = models.columns[cols_lower.index("rmse")]
			fallback = models.sort_values(rmse_col).reset_index()
		else:
			fallback = models.copy()
		print(fallback.head(15).to_string(index=False))
		fallback.to_csv(out_fallback, index=False)
		print(f"Fallback salvo em {out_fallback}")

	# === 10) Log do que falhou (opcional) ===
	if bad:
		print("\nModelos ignorados / problemas detectados:")
		for m, why in bad[:10]:
			print(f"- {m}: {why}")
		if len(bad) > 10:
			print(f"... (+{len(bad)-10} outros)")


if __name__ == "__main__":
	main()

