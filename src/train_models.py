"""
Treino e avaliação dos modelos (linha de comandos)
==================================================
Previsão da direção diária do BTC/USD (classificação binária).

Pontos-chave metodológicos:
  * Divisão TEMPORAL treino/teste (nunca aleatória), para não usar o futuro no treino.
  * Validação cruzada walk-forward (TimeSeriesSplit com intervalo de 1 dia) na afinação.
  * O melhor modelo é escolhido pela validação cruzada; o teste só serve para a avaliação final.
  * Normalização ajustada apenas no treino (dentro de um Pipeline).
  * Comparação com baselines ingénuos (classe maioritária e persistência) e mini-backtest.

Uso:
    python src/train_models.py

Grava models/metrics.json (referência do relatório) e um .joblib por modelo.
"""
import os

import pandas as pd

from modeling import MODEL_NAMES, save_artifacts, train_and_eval

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATS_CSV = os.path.join(BASE, "data", "btc_daily_features.csv")
MODELS_DIR = os.path.join(BASE, "models")
TEST_FRAC = 0.20


def main() -> None:
    feats = pd.read_csv(FEATS_CSV, index_col=0, parse_dates=True)
    fitted, meta = train_and_eval(feats, test_frac=TEST_FRAC)
    save_artifacts(fitted, meta, MODELS_DIR)

    print("Treino: %d dias (%s -> %s)" % (meta["n_train"], *meta["train_period"]))
    print("Teste : %d dias (%s -> %s)" % (meta["n_test"], *meta["test_period"]))
    print()
    print("modelo".ljust(26) + "CV AUC".rjust(8) + "acc".rjust(8) + "f1".rjust(8) + "AUC".rjust(8))
    for name in ["baseline_maioria", "baseline_persistencia"] + MODEL_NAMES:
        r = meta["results"][name]
        cv = r.get("cv_roc_auc")
        print(name.ljust(26) + (format(cv, ".3f") if cv is not None else "-").rjust(8)
              + format(r["accuracy"], ".3f").rjust(8) + format(r["f1"], ".3f").rjust(8)
              + format(r["roc_auc"], ".3f").rjust(8))
    print()
    print("Melhor modelo (escolhido pela validação cruzada): %s" % meta["best_model"])
    print("[OK] Resultados em %s" % os.path.join(MODELS_DIR, "metrics.json"))


if __name__ == "__main__":
    main()
