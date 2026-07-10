"""
Treino e avaliacao de modelos
=============================
Previsao da direcao diaria do BTC/USD (classificacao binaria).

Pontos-chave metodologicos:
  * Divisao TEMPORAL treino/teste (nunca aleatoria) -> evita look-ahead bias.
  * Validacao cruzada com TimeSeriesSplit (walk-forward) na selecao de modelos.
  * Standardizacao ajustada SO no treino (evita fuga de informacao do teste).
  * Comparacao contra baselines ingenuos (classe maioritaria e persistencia).
  * Metricas de classificacao binaria + mini-backtest economico.

Treina 3 modelos (scikit-learn): Logistic Regression, Random Forest, HistGradientBoosting.
"""
import os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

BASE = "/sessions/festive-modest-planck/mnt/TP - ER (Melhoria)/projeto"
DATA = os.path.join(BASE, "data", "btc_daily_features.csv")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
TEST_FRAC = 0.20
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    feature_cols = [c for c in df.columns if c not in
                    ("Open", "High", "Low", "Close", "Volume", "target")]
    return df, df[feature_cols].astype("float64"), df["target"].astype(int), feature_cols


def temporal_split(X, y):
    cut = int(len(X) * (1 - TEST_FRAC))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:], cut


def evaluate(name, y_true, y_pred, y_proba):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    print("\n=== %s ===" % name)
    print("  Accuracy : %.4f" % m["accuracy"])
    print("  Precision: %.4f   Recall: %.4f   F1: %.4f" % (m["precision"], m["recall"], m["f1"]))
    print("  ROC-AUC  : %.4f" % m["roc_auc"])
    print("  Matriz confusao [[TN,FP],[FN,TP]]: %s" % m["confusion_matrix"])
    return m


def backtest(close, y_pred, idx):
    ret_next = (close.shift(-1) / close - 1.0).loc[idx].fillna(0.0).values
    strat = np.where(y_pred == 1, ret_next, 0.0)
    return {"retorno_estrategia": float(np.prod(1 + strat) - 1),
            "retorno_buy_and_hold": float(np.prod(1 + ret_next) - 1)}


def main():
    df, X, y, feature_cols = load_data()
    X_tr, X_te, y_tr, y_te, cut = temporal_split(X, y)
    print("Treino: %d dias (%s -> %s)" % (len(X_tr), df.index[0].date(), df.index[cut-1].date()))
    print("Teste : %d dias (%s -> %s)" % (len(X_te), df.index[cut].date(), df.index[-1].date()))

    tscv = TimeSeriesSplit(n_splits=5)
    results, fitted = {}, {}

    print("\n############ BASELINES ############")
    maj = int(y_tr.mode()[0])
    results["baseline_maioria"] = evaluate("Baseline: classe maioritaria", y_te,
                                           np.full(len(y_te), maj),
                                           np.full(len(y_te), maj, dtype=float))
    yp_pers = (X_te["return_1d"].values > 0).astype(int)
    results["baseline_persistencia"] = evaluate("Baseline: persistencia", y_te,
                                                yp_pers, yp_pers.astype(float))

    configs = {
        "logistic_regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [3, 5, 8], "min_samples_leaf": [10, 30]}),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1],
             "max_iter": [200, 400], "l2_regularization": [0.0, 1.0]}),
    }

    print("\n############ MODELOS ############")
    for name, (estimator, grid) in configs.items():
        print("\n>>> A treinar %s (GridSearchCV walk-forward) ..." % name)
        gs = GridSearchCV(estimator, grid, cv=tscv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        print("    Melhores params: %s" % gs.best_params_)
        print("    ROC-AUC CV (treino): %.4f" % gs.best_score_)
        y_pred = best.predict(X_te)
        y_proba = best.predict_proba(X_te)[:, 1]
        m = evaluate(name, y_te, y_pred, y_proba)
        m["cv_roc_auc"] = float(gs.best_score_)
        m["best_params"] = gs.best_params_
        m["backtest"] = backtest(df["Close"], y_pred, X_te.index)
        results[name], fitted[name] = m, best

    model_names = list(configs.keys())
    best_name = max(model_names, key=lambda n: (round(results[n]["roc_auc"], 4),
                                                round(results[n]["f1"], 4)))
    print("\n############ MELHOR MODELO: %s (ROC-AUC teste = %.4f) ############"
          % (best_name, results[best_name]["roc_auc"]))

    for name, model in fitted.items():
        joblib.dump(model, os.path.join(MODELS_DIR, name + ".joblib"))

    meta = {
        "feature_cols": feature_cols, "best_model": best_name, "test_frac": TEST_FRAC,
        "n_train": len(X_tr), "n_test": len(X_te),
        "train_period": [str(df.index[0].date()), str(df.index[cut-1].date())],
        "test_period": [str(df.index[cut].date()), str(df.index[-1].date())],
        "target_pos_rate": float(y.mean()), "results": results,
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.joblib"))
    print("\n[OK] Modelos + metrics.json gravados em %s" % MODELS_DIR)

    print("\n===== RESUMO (holdout) =====")
    print("modelo".ljust(28) + "acc".rjust(8) + "f1".rjust(8) + "roc_auc".rjust(10))
    for n in ["baseline_maioria", "baseline_persistencia"] + model_names:
        r = results[n]
        print(n.ljust(28) + format(r["accuracy"], ".3f").rjust(8)
              + format(r["f1"], ".3f").rjust(8) + format(r["roc_auc"], ".3f").rjust(10))


if __name__ == "__main__":
    main()
