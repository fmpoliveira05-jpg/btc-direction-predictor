import threading

import numpy as np
import pytest

import modeling
from features import build_features


@pytest.fixture
def feats(daily):
    return build_features(daily)


def test_divisao_temporal_mantem_a_ordem(feats):
    cols, Xtr, Xte, ytr, yte, cut = modeling.split_xy(feats, 0.2)
    assert Xtr.index.max() < Xte.index.min(), "o teste tem de ser sempre posterior ao treino"
    assert len(Xtr) + len(Xte) == len(feats)
    assert abs(len(Xte) / len(feats) - 0.2) < 0.01


def test_validacao_cruzada_deixa_um_dia_de_intervalo(feats):
    X = feats[modeling.feature_columns(feats)]
    for train_idx, val_idx in modeling.time_series_cv().split(X):
        assert val_idx.min() - train_idx.max() == 2


@pytest.mark.parametrize("name", modeling.MODEL_NAMES)
def test_cada_familia_treina_e_devolve_metricas(feats, name):
    model, metrics, meta = modeling.train_single(name, feats, {}, test_frac=0.2, do_cv=False)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix", "backtest"]:
        assert key in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert np.array(metrics["confusion_matrix"]).sum() == meta["n_test"]
    assert model.predict_proba(feats[metrics["feature_cols"]].iloc[:3]).shape == (3, 2)


def test_selecao_de_features_e_respeitada(feats):
    cols = ["return_1d", "rsi_14"]
    model, metrics, _ = modeling.train_single("logistic_regression", feats, {}, do_cv=False, cols=cols)
    assert metrics["feature_cols"] == cols


def test_roc_auc_com_uma_so_classe_nao_rebenta():
    assert np.isnan(modeling._roc_auc(np.ones(5), np.linspace(0, 1, 5)))


def test_automl_devolve_o_melhor_da_tabela(feats):
    fam, model, metrics, meta, board = modeling.automl(feats, families=["logistic_regression"], n_configs=2)
    assert fam == "logistic_regression"
    assert metrics["cv_roc_auc"] == board[0]["cv_roc_auc"]
    assert board == sorted(board, key=lambda r: r["cv_roc_auc"], reverse=True)


def test_automl_pode_ser_interrompido(feats):
    stop = threading.Event()
    stop.set()
    result = modeling.automl_stoppable(feats, stop_event=stop)
    assert result[:4] == (None, None, None, None)


def test_registo_de_versoes(tmp_path, feats):
    model, metrics, meta = modeling.train_single("logistic_regression", feats, {}, do_cv=False)
    first = modeling.save_run(str(tmp_path), "logistic_regression", model, metrics, meta, feats, 1.0)
    second = modeling.save_run(str(tmp_path), "logistic_regression", model, metrics, meta, feats, 1.0)

    assert (first, second) == ("run_0001", "run_0002")
    assert [r["id"] for r in modeling.list_runs(str(tmp_path))] == ["run_0001", "run_0002"]
    loaded = modeling.load_run(str(tmp_path), second)
    assert loaded.predict(feats[metrics["feature_cols"]].iloc[:1]).shape == (1,)

    modeling.reset_registry(str(tmp_path))
    assert modeling.list_runs(str(tmp_path)) == []


def test_registo_corrompido_nao_impede_o_arranque(tmp_path):
    (tmp_path / "registry.json").write_text("{isto não é json")
    assert modeling.load_registry(str(tmp_path))["runs"] == []
