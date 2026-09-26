"""
Treina os três modelos de demonstração com os hiperparâmetros por omissão, pelo mesmo
caminho da aplicação (curva de aprendizagem, treino, validação cruzada e baselines), e
guarda-os em demo/modelos/ para a versão web arrancar já com modelos treinados.

Tem de correr com as mesmas versões do scikit-learn, NumPy, pandas e joblib que o Pyodide
usado pela demonstração (ver demo/requirements-pretreino.txt); caso contrário os ficheiros
.joblib podem não abrir no browser.

    python demo/pretreinar.py
"""
import os
import sys
import time

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
from modeling import (MODEL_NAMES, DEFAULT_PARAMS, train_single, compute_baselines,  # noqa: E402
                      learning_curve_temporal, save_run)

DESTINO = os.path.join(BASE, "demo", "modelos")
TEST_FRAC = 0.20


def main():
    feats = pd.read_csv(os.path.join(BASE, "data", "btc_daily_features.csv"),
                        index_col=0, parse_dates=True)
    os.makedirs(DESTINO, exist_ok=True)
    for f in os.listdir(DESTINO):
        os.remove(os.path.join(DESTINO, f))
    base = compute_baselines(feats, TEST_FRAC)
    ds = {"n_total": len(feats), "last_date": str(feats.index[-1].date()),
          "source": "pre-trained for the web demo"}
    for name in MODEL_NAMES:
        params = dict(DEFAULT_PARAMS[name])
        t0 = time.time()
        lc = learning_curve_temporal(name, feats, params, TEST_FRAC, n_points=7)
        model, m, meta = train_single(name, feats, params, TEST_FRAC, do_cv=True)
        rid = save_run(DESTINO, name, model, m, meta, feats, time.time() - t0,
                       baselines=base, dataset_info=ds, lc=lc)
        print("%s %s ROC-AUC teste %.3f, CV %.3f" % (rid, name, m["roc_auc"], m["cv_roc_auc"]))


if __name__ == "__main__":
    main()
