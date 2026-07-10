"""
Treino, avaliacao, AutoML e (de)serializacao de modelos (modulo reutilizavel)
=============================================================================
Usado pelo script de treino inicial (train_models.py) e pela aplicacao Streamlit
(treino interativo, com hiperparametros ajustaveis por modelo, e AutoML).

Tres familias de modelos, de complexidade crescente:
  * LEVE        -> Regressao Logistica   (linear, instantaneo)
  * INTERMEDIO  -> Random Forest         (ensemble por bagging)
  * PESADO      -> HistGradientBoosting  (gradient boosting)
"""
import os, json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from features import feature_columns

RANDOM_STATE = 42
MODEL_NAMES = ["logistic_regression", "random_forest", "hist_gradient_boosting"]

# Metadados de cada familia (rotulo, "peso" e descricao para a interface)
MODEL_REGISTRY = {
    "logistic_regression": {
        "label": "Light - Logistic Regression", "weight": "Light",
        "desc": "Linear model, very fast and interpretable. A solid baseline."},
    "random_forest": {
        "label": "Intermediate - Random Forest", "weight": "Intermediate",
        "desc": "Ensemble of decision trees (bagging). Captures non-linear patterns; moderate training time."},
    "hist_gradient_boosting": {
        "label": "Heavy - HistGradientBoosting", "weight": "Heavy",
        "desc": "Histogram-based gradient boosting. The most sophisticated of the three."},
}

# Hiperparametros por defeito (servem de valor inicial nos widgets da app)
DEFAULT_PARAMS = {
    "logistic_regression": {"C": 0.1, "class_weight": "balanced",
                            "solver": "lbfgs", "max_iter": 2000},
    "random_forest": {"n_estimators": 400, "max_depth": 5, "min_samples_leaf": 30,
                      "max_features": "sqrt", "class_weight": "balanced"},
    "hist_gradient_boosting": {"learning_rate": 0.03, "max_depth": 2, "max_iter": 200,
                               "l2_regularization": 0.0, "min_samples_leaf": 20},
}

# Espaco de procura usado pelo AutoML (amostragem aleatoria)
def _space(name, r):
    if name == "logistic_regression":
        return {"C": float(10 ** r.uniform(-3, 2)),
                "class_weight": r.choice(["balanced", None]),
                "solver": "lbfgs", "max_iter": 2000}
    if name == "random_forest":
        return {"n_estimators": r.choice([200, 300, 400]),
                "max_depth": r.choice([3, 4, 5, 6, 8]),
                "min_samples_leaf": r.choice([5, 10, 20, 30, 50]),
                "max_features": r.choice(["sqrt", "log2"]),
                "class_weight": "balanced"}
    return {"learning_rate": r.choice([0.01, 0.03, 0.05, 0.1]),
            "max_depth": r.choice([2, 3, 4, 5]),
            "max_iter": r.choice([150, 200, 300, 400]),
            "l2_regularization": r.choice([0.0, 0.5, 1.0, 2.0]),
            "min_samples_leaf": r.choice([10, 20, 30, 50])}


# ----------------------------------------------------------------------
# Construcao de estimadores
# ----------------------------------------------------------------------
def _norm_cw(v):
    return None if v in (None, "none", "None", "") else v


def make_model(name, params=None):
    """Cria um estimador scikit-learn com os hiperparametros dados."""
    p = dict(DEFAULT_PARAMS[name], **(params or {}))
    if name == "logistic_regression":
        clf = LogisticRegression(C=float(p["C"]), max_iter=int(p["max_iter"]),
                                 solver=p["solver"], class_weight=_norm_cw(p["class_weight"]))
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "random_forest":
        md = p["max_depth"]; md = None if not md else int(md)
        return RandomForestClassifier(
            n_estimators=int(p["n_estimators"]), max_depth=md,
            min_samples_leaf=int(p["min_samples_leaf"]), max_features=p["max_features"],
            class_weight=_norm_cw(p["class_weight"]), random_state=RANDOM_STATE, n_jobs=-1)
    md = p["max_depth"]; md = None if not md else int(md)
    return HistGradientBoostingClassifier(
        learning_rate=float(p["learning_rate"]), max_depth=md, max_iter=int(p["max_iter"]),
        l2_regularization=float(p["l2_regularization"]),
        min_samples_leaf=int(p["min_samples_leaf"]), random_state=RANDOM_STATE)


# ----------------------------------------------------------------------
# Avaliacao
# ----------------------------------------------------------------------
def _metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _backtest(close, y_pred, idx):
    ret = (close.shift(-1) / close - 1.0).loc[idx].fillna(0.0).values
    strat = np.where(np.asarray(y_pred) == 1, ret, 0.0)
    return {"retorno_estrategia": float(np.prod(1 + strat) - 1),
            "retorno_buy_and_hold": float(np.prod(1 + ret) - 1)}


def split_xy(feats, test_frac, cols=None):
    cols = list(cols) if cols else feature_columns(feats)
    X = feats[cols].astype("float64")
    y = feats["target"].astype(int)
    cut = int(len(X) * (1 - test_frac))
    return cols, X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:], cut


def compute_baselines(feats, test_frac=0.20):
    cols, Xtr, Xte, ytr, yte, cut = split_xy(feats, test_frac)
    maj = int(ytr.mode()[0])
    b1 = _metrics(yte, np.full(len(yte), maj), np.full(len(yte), maj, dtype=float))
    yp = (Xte["return_1d"].values > 0).astype(int)
    b2 = _metrics(yte, yp, yp.astype(float))
    return {"baseline_maioria": b1, "baseline_persistencia": b2}


def _period_meta(feats, cut, test_frac):
    return {"n_train": int(cut), "n_test": int(len(feats) - cut), "test_frac": test_frac,
            "train_period": [str(feats.index[0].date()), str(feats.index[cut - 1].date())],
            "test_period": [str(feats.index[cut].date()), str(feats.index[-1].date())]}


def train_single(name, feats, params, test_frac=0.20, do_cv=True, cols=None):
    """Treina UM modelo com hiperparametros especificos. Devolve (model, metrics, meta)."""
    cols, Xtr, Xte, ytr, yte, cut = split_xy(feats, test_frac, cols)
    model = make_model(name, params)
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    pr = model.predict_proba(Xte)[:, 1]
    m = _metrics(yte, yp, pr)
    m["cv_roc_auc"] = float("nan")
    if do_cv:
        try:
            m["cv_roc_auc"] = float(np.mean(cross_val_score(
                make_model(name, params), Xtr, ytr,
                cv=TimeSeriesSplit(n_splits=5), scoring="roc_auc", n_jobs=-1)))
        except Exception:
            pass
    m["params"] = params
    m["feature_cols"] = list(cols)
    m["backtest"] = _backtest(feats["Close"], yp, Xte.index)
    return model, m, _period_meta(feats, cut, test_frac)


def automl(feats, test_frac=0.20, families=None, n_configs=6, seed=42, progress=None):
    """Procura aleatoria de configuracoes por familia. Devolve melhor + leaderboard."""
    import random
    families = families or list(MODEL_NAMES)
    rnd = random.Random(seed)
    cols, Xtr, Xte, ytr, yte, cut = split_xy(feats, test_frac, cols)
    tscv = TimeSeriesSplit(n_splits=5)
    leaderboard = []
    total = max(1, len(families) * n_configs)
    done = 0
    for fam in families:
        for _ in range(n_configs):
            params = _space(fam, rnd)
            try:
                cv = float(np.mean(cross_val_score(make_model(fam, params), Xtr, ytr,
                                                   cv=tscv, scoring="roc_auc", n_jobs=-1)))
            except Exception:
                cv = float("nan")
            leaderboard.append({"familia": fam, "label": MODEL_REGISTRY[fam]["weight"],
                                "cv_roc_auc": cv, "params": params})
            done += 1
            if progress:
                progress(done / total, fam)
    valid = [x for x in leaderboard if x["cv_roc_auc"] == x["cv_roc_auc"]]
    valid.sort(key=lambda d: d["cv_roc_auc"], reverse=True)
    best = valid[0]
    model, m, meta = train_single(best["familia"], feats, best["params"], test_frac,
                                  do_cv=False, cols=cols)
    m["cv_roc_auc"] = best["cv_roc_auc"]
    return best["familia"], model, m, meta, valid


# ----------------------------------------------------------------------
# Persistencia (registry.json + artefactos .joblib)
# ----------------------------------------------------------------------
def _reg_path(models_dir):
    return os.path.join(models_dir, "registry.json")


def load_registry(models_dir):
    p = _reg_path(models_dir)
    default = {"runs": [], "baselines": {}, "dataset": {}, "feature_cols": []}
    if os.path.exists(p):
        try:
            with open(p) as f:
                reg = json.load(f)
            reg.setdefault("runs", [])
            return reg
        except (json.JSONDecodeError, ValueError):
            return default
    return default


def save_trained(models_dir, name, model, metrics, meta, feats,
                 dataset_info=None, baselines=None):
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, name + ".joblib"))
    reg = load_registry(models_dir)
    reg["feature_cols"] = feature_columns(feats)
    reg["models"][name] = {
        "label": MODEL_REGISTRY[name]["label"], "weight": MODEL_REGISTRY[name]["weight"],
        "metrics": metrics, "meta": meta,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if baselines is not None:
        reg["baselines"] = baselines
    if dataset_info is not None:
        reg["dataset"] = dataset_info
    with open(_reg_path(models_dir), "w") as f:
        json.dump(reg, f, indent=2, default=str)
    return reg


def load_model(models_dir, name):
    return joblib.load(os.path.join(models_dir, name + ".joblib"))


# ----------------------------------------------------------------------
# Treino completo (CLI) - usado por train_models.py
# ----------------------------------------------------------------------
GRIDS = {
    "logistic_regression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "random_forest": {"n_estimators": [200, 400], "max_depth": [3, 5, 8],
                      "min_samples_leaf": [10, 30]},
    "hist_gradient_boosting": {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1],
                               "max_iter": [200, 400], "l2_regularization": [0.0, 1.0]},
}


def train_and_eval(feats, test_frac=0.20, quick=False):
    cols, Xtr, Xte, ytr, yte, cut = split_xy(feats, test_frac)
    tscv = TimeSeriesSplit(n_splits=5)
    results = compute_baselines(feats, test_frac)
    fitted = {}
    base_estimators = {"logistic_regression": Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
                       "random_forest": RandomForestClassifier(
                           random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
                       "hist_gradient_boosting": HistGradientBoostingClassifier(
                           random_state=RANDOM_STATE)}
    for name in MODEL_NAMES:
        gs = GridSearchCV(base_estimators[name], GRIDS[name], cv=tscv,
                          scoring="roc_auc", n_jobs=-1)
        gs.fit(Xtr, ytr)
        model = gs.best_estimator_
        yp = model.predict(Xte); pr = model.predict_proba(Xte)[:, 1]
        m = _metrics(yte, yp, pr)
        m["cv_roc_auc"] = float(gs.best_score_)
        m["best_params"] = gs.best_params_
        m["backtest"] = _backtest(feats["Close"], yp, Xte.index)
        results[name], fitted[name] = m, model
    best = max(MODEL_NAMES, key=lambda n: (round(results[n]["roc_auc"], 4),
                                           round(results[n]["f1"], 4)))
    meta = {"feature_cols": cols, "best_model": best, "test_frac": test_frac,
            "n_train": len(Xtr), "n_test": len(Xte), "n_total": len(feats),
            "train_period": [str(feats.index[0].date()), str(feats.index[cut - 1].date())],
            "test_period": [str(feats.index[cut].date()), str(feats.index[-1].date())],
            "target_pos_rate": float(ytr.mean()), "results": results}
    return fitted, meta


def save_artifacts(fitted, meta, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    for name, model in fitted.items():
        joblib.dump(model, os.path.join(models_dir, name + ".joblib"))
    joblib.dump(meta["feature_cols"], os.path.join(models_dir, "feature_cols.joblib"))
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ======================================================================
# EXTENSOES: learning curve, AutoML com interrupcao, e registo por VERSAO
# (cada treino do utilizador = uma versao carregavel, estilo best.pt)
# ======================================================================
def learning_curve_temporal(name, feats, params, test_frac=0.20, n_points=7,
                            stop_event=None, on_step=None, cols=None):
    """Curva de aprendizagem temporal: treina em janelas crescentes do treino e
    avalia ROC-AUC no proprio treino e no holdout. Suporta interrupcao."""
    cols, Xtr, Xte, ytr, yte, cut = split_xy(feats, test_frac, cols)
    fracs = np.linspace(0.2, 1.0, n_points)
    out = {"frac": [], "n": [], "train_auc": [], "val_auc": []}
    for i, fr in enumerate(fracs):
        if stop_event is not None and stop_event.is_set():
            break
        k = max(60, int(len(Xtr) * fr))
        mdl = make_model(name, params)
        mdl.fit(Xtr.iloc[:k], ytr.iloc[:k])
        tr = float(roc_auc_score(ytr.iloc[:k], mdl.predict_proba(Xtr.iloc[:k])[:, 1]))
        va = float(roc_auc_score(yte, mdl.predict_proba(Xte)[:, 1]))
        out["frac"].append(round(float(fr), 2))
        out["n"].append(int(k))
        out["train_auc"].append(tr)
        out["val_auc"].append(va)
        if on_step:
            on_step((i + 1) / n_points)
    return out


def automl_stoppable(feats, test_frac=0.20, families=None, n_configs=6, seed=42,
                     progress=None, stop_event=None, cols=None):
    """Como automl(), mas verifica stop_event entre configuracoes.
    Devolve (best_family, model, metrics, meta, leaderboard) ou
    (None, None, None, None, leaderboard) se interrompido."""
    import random
    families = families or list(MODEL_NAMES)
    rnd = random.Random(seed)
    cols, Xtr, Xte, ytr, yte, cut = split_xy(feats, test_frac, cols)
    tscv = TimeSeriesSplit(n_splits=5)
    leaderboard = []
    total = max(1, len(families) * n_configs)
    done = 0
    for fam in families:
        for _ in range(n_configs):
            if stop_event is not None and stop_event.is_set():
                return None, None, None, None, leaderboard
            params = _space(fam, rnd)
            try:
                cv = float(np.mean(cross_val_score(make_model(fam, params), Xtr, ytr,
                                                   cv=tscv, scoring="roc_auc", n_jobs=-1)))
            except Exception:
                cv = float("nan")
            leaderboard.append({"familia": fam, "label": MODEL_REGISTRY[fam]["weight"],
                                "cv_roc_auc": cv, "params": params})
            done += 1
            if progress:
                progress(done / total, fam)
    valid = [x for x in leaderboard if x["cv_roc_auc"] == x["cv_roc_auc"]]
    valid.sort(key=lambda d: d["cv_roc_auc"], reverse=True)
    best = valid[0]
    model, m, meta = train_single(best["familia"], feats, best["params"], test_frac,
                                  do_cv=False, cols=cols)
    m["cv_roc_auc"] = best["cv_roc_auc"]
    return best["familia"], model, m, meta, valid


# ---- Registo por versao (runs) ----
def list_runs(models_dir):
    return load_registry(models_dir).get("runs", [])


def _next_run_id(reg):
    return "run_%04d" % (len(reg.get("runs", [])) + 1)


def save_run(models_dir, family, model, metrics, meta, feats, train_time,
             baselines=None, dataset_info=None, lc=None):
    """Guarda UMA versao treinada (joblib + entrada no registry)."""
    os.makedirs(models_dir, exist_ok=True)
    reg = load_registry(models_dir)
    reg.setdefault("runs", [])
    rid = _next_run_id(reg)
    joblib.dump(model, os.path.join(models_dir, rid + ".joblib"))
    reg["feature_cols"] = feature_columns(feats)
    if baselines is not None:
        reg["baselines"] = baselines
    if dataset_info is not None:
        reg["dataset"] = dataset_info
    reg["runs"].append({
        "id": rid, "family": family, "label": MODEL_REGISTRY[family]["label"],
        "weight": MODEL_REGISTRY[family]["weight"], "params": metrics.get("params", {}),
        "feature_cols": metrics.get("feature_cols") or feature_columns(feats),
        "metrics": metrics, "meta": meta, "lc": lc,
        "train_time_s": round(float(train_time), 2),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    with open(_reg_path(models_dir), "w") as f:
        json.dump(reg, f, indent=2, default=str)
    return rid


def load_run(models_dir, run_id):
    return joblib.load(os.path.join(models_dir, run_id + ".joblib"))


def reset_registry(models_dir, keep_dataset=True):
    """Limpa todas as versoes treinadas (mantem info do dataset se existir)."""
    reg = load_registry(models_dir)
    new = {"runs": [], "baselines": {}, "dataset": reg.get("dataset", {}) if keep_dataset else {},
           "feature_cols": reg.get("feature_cols", [])}
    with open(_reg_path(models_dir), "w") as f:
        json.dump(new, f, indent=2, default=str)
