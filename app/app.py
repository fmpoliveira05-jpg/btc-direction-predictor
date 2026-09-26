"""
BTC Direction Predictor - Streamlit application
===============================================
End-to-end workflow, entirely from the interface:
  1. [Data]        download the dataset (Kaggle) and inspect the data-processing /
                   feature-engineering pipeline; optionally select feature groups
  2. [Training]    train models with adjustable hyperparameters OR AutoML
                   (live learning curve, elapsed-time counter and a Stop button)
  3. [Prediction]  get the BTC next-day direction with a model you trained
  4. [Comparison]  compare the models before training (characteristics) and after
                   training (hyperparameters + metrics, with up/down indicators)
  5. [History]     review predictions made and the realised hit rate

Locally, only models trained by the user are usable for prediction (no pre-trained
models are shipped). The web demo (demo/index.html) starts with the three models
pre-trained by demo/pretreinar.py. Each training run is saved as a new, independently
loadable version.
"""
import os, sys, time, threading
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
from features import (build_features, feature_columns, resample_daily,  # noqa: E402
                      FEATURE_GROUPS, FEATURE_DESCRIPTIONS)
from modeling import (MODEL_NAMES, MODEL_REGISTRY, DEFAULT_PARAMS,  # noqa: E402
                      train_single, compute_baselines, learning_curve_temporal,
                      automl_stoppable, list_runs, save_run, load_run, load_registry)

DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
OHLCV_CSV = os.path.join(DATA_DIR, "btc_daily_ohlcv.csv")
FEATS_CSV = os.path.join(DATA_DIR, "btc_daily_features.csv")
RAW_CSV = os.path.join(DATA_DIR, "btcusd_1-min_data.csv")
HISTORY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.csv")
KAGGLE_DS = "mczielinski/bitcoin-historical-data"
LABELS = {1: "UP", 0: "DOWN"}
# True in the web demo (Stlite: Streamlit running in the browser on Pyodide/WebAssembly).
# There are no threads there, so training runs in the foreground, and Kaggle cannot be reached.
IN_BROWSER = sys.platform == "emscripten"

st.set_page_config(page_title="BTC Direction Predictor", layout="wide")


# ----------------------------------------------------------------------
# Loading / helpers
# ----------------------------------------------------------------------
@st.cache_data
def load_ohlcv():
    return pd.read_csv(OHLCV_CSV, index_col=0, parse_dates=True)


@st.cache_data
def load_feats():
    return pd.read_csv(FEATS_CSV, index_col=0, parse_dates=True)


def data_ready():
    return os.path.exists(OHLCV_CSV) and os.path.exists(FEATS_CSV)


def selected_cols():
    groups = st.session_state.get("sel_groups", list(FEATURE_GROUPS))
    cols = []
    for g in groups:
        cols += FEATURE_GROUPS.get(g, [])
    return cols or [c for g in FEATURE_GROUPS.values() for c in g]


def append_history(row):
    df = pd.DataFrame([row])
    header = not (os.path.exists(HISTORY_CSV) and os.path.getsize(HISTORY_CSV) > 0)
    df.to_csv(HISTORY_CSV, mode="a", header=header, index=False)


def read_history():
    cols = ["timestamp", "model", "base_date", "predicted_date", "prob_up", "prediction"]
    if os.path.exists(HISTORY_CSV) and os.path.getsize(HISTORY_CSV) > 0:
        try:
            return pd.read_csv(HISTORY_CSV)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=cols)
    return pd.DataFrame(columns=cols)


def process_raw_to_data(csv_path):
    from data_pipeline import load_minute_data  # só existe na versão local (não na demonstração)
    daily = resample_daily(load_minute_data(csv_path))
    feats = build_features(daily, with_target=True)
    daily.to_csv(OHLCV_CSV)
    feats.to_csv(FEATS_CSV)
    return len(daily), len(feats)


def append_daily_rows(new_df):
    """Append new daily OHLCV rows to the dataset and rebuild the features."""
    base = pd.read_csv(OHLCV_CSV, index_col=0, parse_dates=True)
    if new_df.index.tz is None:
        new_df.index = new_df.index.tz_localize("UTC")
    new_df = new_df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    combined = pd.concat([base, new_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    feats = build_features(combined, with_target=True)
    combined.to_csv(OHLCV_CSV)
    feats.to_csv(FEATS_CSV)
    return len(combined), len(feats)


# ----------------------------------------------------------------------
# Hyperparameter widgets
# ----------------------------------------------------------------------
def params_widgets(name):
    d = DEFAULT_PARAMS[name]
    if name == "logistic_regression":
        c1, c2 = st.columns(2)
        C = c1.number_input("C", 0.001, 100.0, float(d["C"]), format="%.3f")
        cw = c2.selectbox("class_weight", ["balanced", "none"], 0)
        mi = c1.number_input("max_iter", 200, 5000, int(d["max_iter"]), step=100)
        sv = c2.selectbox("solver", ["lbfgs", "liblinear", "saga"], 0)
        return {"C": C, "class_weight": cw, "solver": sv, "max_iter": mi}
    if name == "random_forest":
        c1, c2 = st.columns(2)
        n = c1.slider("n_estimators", 50, 800, int(d["n_estimators"]), 50)
        md = c2.slider("max_depth (0 = no limit)", 0, 20, int(d["max_depth"]))
        leaf = c1.slider("min_samples_leaf", 1, 60, int(d["min_samples_leaf"]))
        mf = c2.selectbox("max_features", ["sqrt", "log2", "none"], 0)
        cw = c1.selectbox("class_weight", ["balanced", "balanced_subsample", "none"], 0)
        return {"n_estimators": n, "max_depth": md, "min_samples_leaf": leaf,
                "max_features": (None if mf == "none" else mf), "class_weight": cw}
    c1, c2 = st.columns(2)
    lr = c1.number_input("learning_rate", 0.005, 0.5, float(d["learning_rate"]), format="%.3f")
    md = c2.slider("max_depth (0 = no limit)", 0, 12, int(d["max_depth"]))
    mit = c1.slider("max_iter", 50, 1000, int(d["max_iter"]), 50)
    l2 = c2.number_input("l2_regularization", 0.0, 10.0, float(d["l2_regularization"]), format="%.2f")
    leaf = c1.slider("min_samples_leaf", 5, 100, int(d["min_samples_leaf"]))
    return {"learning_rate": lr, "max_depth": md, "max_iter": mit,
            "l2_regularization": l2, "min_samples_leaf": leaf}


def learning_curve_fig(lc):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(lc["n"], lc["train_auc"], "o-", label="Train ROC-AUC", color="#2a5d9c")
    ax.plot(lc["n"], lc["val_auc"], "s-", label="Validation ROC-AUC", color="#f7931a")
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_xlabel("Training samples"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Learning curve"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def show_result(name, m, meta, base):
    c = st.columns(4)
    c[0].metric("Accuracy", "%.3f" % m["accuracy"],
                delta="%.3f vs majority" % (m["accuracy"] - base["baseline_maioria"]["accuracy"]))
    c[1].metric("F1", "%.3f" % m["f1"])
    c[2].metric("ROC-AUC", "%.3f" % m["roc_auc"])
    cv = m.get("cv_roc_auc", float("nan"))
    c[3].metric("ROC-AUC (CV)", "-" if cv != cv else "%.3f" % cv)
    bt = m["backtest"]
    st.caption("Backtest on the test set (%s to %s): strategy %.1f%% vs buy & hold %.1f%%."
               % (meta["test_period"][0], meta["test_period"][1],
                  bt["retorno_estrategia"] * 100, bt["retorno_buy_and_hold"] * 100))


# ----------------------------------------------------------------------
# Background training job
# ----------------------------------------------------------------------
def _worker_manual(shared, stop, feats, name, params, test_frac, do_cv, cols):
    t0 = time.time()
    try:
        lc = learning_curve_temporal(
            name, feats, params, test_frac, n_points=7, stop_event=stop, cols=cols,
            on_step=lambda f: shared.update(progress=0.05 + 0.7 * f,
                                            stage="Computing learning curve"))
        if stop.is_set():
            shared.update(status="stopped", train_time=time.time() - t0); return
        shared.update(progress=0.8, stage="Fitting final model")
        model, m, meta = train_single(name, feats, params, test_frac, do_cv, cols=cols)
        if stop.is_set():
            shared.update(status="stopped", train_time=time.time() - t0); return
        shared.update(progress=0.95, stage="Evaluating baselines")
        base = compute_baselines(feats, test_frac)
        shared["result"] = {"name": name, "model": model, "metrics": m, "meta": meta,
                            "baselines": base, "lc": lc}
        shared.update(progress=1.0, stage="Done", status="done", train_time=time.time() - t0)
    except Exception as e:
        shared.update(status="error", error=str(e), train_time=time.time() - t0)


def _worker_automl(shared, stop, feats, families, n_cfg, test_frac, cols):
    t0 = time.time()
    try:
        def prog(frac, fam):
            shared.update(progress=frac, stage="AutoML: testing %s" % MODEL_REGISTRY[fam]["weight"])
        best, model, m, meta, lb = automl_stoppable(
            feats, test_frac, families, n_cfg, progress=prog, stop_event=stop, cols=cols)
        if best is None:
            shared.update(status="stopped", train_time=time.time() - t0); return
        base = compute_baselines(feats, test_frac)
        shared["result"] = {"name": best, "model": model, "metrics": m, "meta": meta,
                            "baselines": base, "lc": None, "leaderboard": lb}
        shared.update(progress=1.0, stage="Done", status="done", train_time=time.time() - t0)
    except Exception as e:
        shared.update(status="error", error=str(e), train_time=time.time() - t0)


def _launch(target, *args):
    shared = {"status": "running", "progress": 0.0, "stage": "Starting"}
    stop = threading.Event()
    if IN_BROWSER:
        # Pyodide has no threads: train in the foreground and show the result straight away.
        with st.spinner("Training in the browser (the Random Forest can take up to a minute)..."):
            target(shared, stop, *args)
        _consume_job({"shared": shared}, args[0])
        st.rerun()
    th = threading.Thread(target=target, args=(shared, stop) + args, daemon=True)
    th.start()
    st.session_state["job"] = {"thread": th, "shared": shared, "stop": stop, "start": time.time()}
    st.rerun()


def _render_running(job):
    sh = job["shared"]
    st.info("Training in progress.")
    c1, c2 = st.columns(2)
    c1.metric("Elapsed time", "%.1f s" % (time.time() - job["start"]))
    c2.metric("Stage", sh.get("stage", "..."))
    st.progress(min(float(sh.get("progress", 0.0)), 1.0))
    if st.button("Stop training"):
        job["stop"].set()
        sh["stage"] = "Stopping..."


def _consume_job(job, feats):
    sh = job["shared"]
    status = sh.get("status")
    if status == "done":
        res = sh["result"]
        ds = {"n_total": len(feats), "last_date": str(feats.index[-1].date()), "source": "trained in app"}
        rid = save_run(MODELS_DIR, res["name"], res["model"], res["metrics"], res["meta"],
                       feats, sh.get("train_time", 0.0), baselines=res["baselines"],
                       dataset_info=ds, lc=res.get("lc"))
        st.session_state["last_train"] = {
            "status": "done", "rid": rid, "name": res["name"], "metrics": res["metrics"],
            "meta": res["meta"], "lc": res.get("lc"), "baselines": res["baselines"],
            "train_time": sh.get("train_time", 0.0), "leaderboard": res.get("leaderboard")}
    elif status == "stopped":
        st.session_state["last_train"] = {"status": "stopped"}
    else:
        st.session_state["last_train"] = {"status": "error", "error": sh.get("error", "unknown")}
    st.session_state["job"] = None


def _render_last_train():
    lt = st.session_state.get("last_train")
    if not lt:
        return
    st.markdown("---")
    if lt["status"] == "stopped":
        st.warning("The last training was interrupted. Nothing was saved.")
        return
    if lt["status"] == "error":
        st.error("The last training failed: %s" % lt.get("error"))
        return
    st.subheader("Last training result")
    st.success("Saved as version %s (%s)." % (lt["rid"], MODEL_REGISTRY[lt["name"]]["label"]))
    show_result(lt["name"], lt["metrics"], lt["meta"], lt["baselines"])
    c1, c2 = st.columns([1, 2])
    c1.metric("Training time", "%.1f s" % lt["train_time"])
    if lt.get("lc"):
        c2.pyplot(learning_curve_fig(lt["lc"]))
    if lt.get("leaderboard"):
        st.write("AutoML leaderboard (top 10 by cross-validated ROC-AUC):")
        df = pd.DataFrame([{"Family": MODEL_REGISTRY[x["familia"]]["weight"],
                            "CV ROC-AUC": round(x["cv_roc_auc"], 4),
                            "Params": str(x["params"])} for x in lt["leaderboard"][:10]])
        st.dataframe(df, width="stretch")


def _render_train_controls(feats):
    cols = selected_cols()
    st.caption("Dataset: %d samples (%s to %s) - %d features selected."
               % (len(feats), feats.index[0].date(), feats.index[-1].date(), len(cols)))
    mode = st.radio("Training mode", ["Manual", "AutoML"], horizontal=True)
    test_frac = st.slider("Test set fraction (temporal holdout)", 0.10, 0.30, 0.20, 0.05)
    if mode == "Manual":
        name = st.selectbox("Model", MODEL_NAMES, format_func=lambda n: MODEL_REGISTRY[n]["label"])
        st.info(MODEL_REGISTRY[name]["desc"])
        with st.form("train_form"):
            st.markdown("**Hyperparameters**")
            params = params_widgets(name)
            do_cv = st.checkbox("Compute temporal cross-validation", True)
            submit = st.form_submit_button("Train model", type="primary")
        if submit:
            _launch(_worker_manual, feats, name, params, test_frac, do_cv, cols)
    else:
        st.info("AutoML automatically searches good configurations (random search with "
                "temporal cross-validation) and keeps the best model found.")
        fams = st.multiselect("Model families", MODEL_NAMES, default=MODEL_NAMES,
                              format_func=lambda n: MODEL_REGISTRY[n]["weight"])
        n_cfg = st.slider("Configurations per family", 3, 15, 5)
        if st.button("Run AutoML", type="primary"):
            if not fams:
                st.error("Select at least one family.")
            else:
                _launch(_worker_automl, feats, fams, n_cfg, test_frac, cols)


# ----------------------------------------------------------------------
# Comparison helpers (HTML tables with up/down indicators)
# ----------------------------------------------------------------------
def _cell(val, rank):
    arrow = {"best": '<span style="color:#1b8a4b;font-weight:700">&#9650;</span>',
             "worst": '<span style="color:#c0392b;font-weight:700">&#9660;</span>',
             "mid": '<span style="color:#999">&#9679;</span>', "na": ""}[rank]
    return '<td style="padding:6px 10px;border-bottom:1px solid #ccc;text-align:center">%s %s</td>' % (val, arrow)


def _ranks(values, higher_better=True):
    nums = [v for v in values if v == v]
    if len(set(nums)) <= 1:
        return ["na"] * len(values)
    best = max(nums) if higher_better else min(nums)
    worst = min(nums) if higher_better else max(nums)
    out = []
    for v in values:
        if v != v:
            out.append("na")
        elif v == best:
            out.append("best")
        elif v == worst:
            out.append("worst")
        else:
            out.append("mid")
    return out


def after_training_html(runs):
    labels = ["%s<br><span style='color:#888;font-size:11px'>%s</span>"
              % (MODEL_REGISTRY[r["family"]]["weight"], r["id"]) for r in runs]
    head = ('<th style="padding:6px 10px;text-align:left;border-bottom:2px solid #1f2d3d">'
            'Parameter / metric</th>')
    head += "".join('<th style="padding:6px 10px;text-align:center;border-bottom:2px solid #1f2d3d">%s</th>' % l
                    for l in labels)
    html = '<table style="border-collapse:collapse;width:100%;font-size:13px"><tr>' + head + "</tr>"

    pkeys = ["C", "class_weight", "solver", "max_iter", "n_estimators", "max_depth",
             "min_samples_leaf", "max_features", "learning_rate", "l2_regularization"]
    pkeys = [k for k in pkeys if any(k in r["params"] for r in runs)]
    html += ('<tr><td colspan="%d" style="padding:6px 10px;background:#1f2d3d;color:#fff;'
             'font-weight:700">Hyperparameters used</td></tr>' % (len(runs) + 1))
    for k in pkeys:
        row = '<tr><td style="padding:6px 10px;border-bottom:1px solid #ccc">%s</td>' % k
        for r in runs:
            v = r["params"].get(k, "-")
            row += ('<td style="padding:6px 10px;border-bottom:1px solid #ccc;text-align:center">%s</td>'
                    % ("-" if v is None else v))
        html += row + "</tr>"

    perf = [("Accuracy", "accuracy", True, "pct"), ("F1", "f1", True, "pct"),
            ("ROC-AUC", "roc_auc", True, "pct"), ("ROC-AUC (CV)", "cv_roc_auc", True, "pct"),
            ("Backtest strategy", "backtest", True, "pct"),
            ("Training time", "train_time_s", False, "s")]
    html += ('<tr><td colspan="%d" style="padding:6px 10px;background:#1f2d3d;color:#fff;'
             'font-weight:700">Performance (green up = best, red down = worst)</td></tr>'
             % (len(runs) + 1))
    for lbl, key, hb, fmt in perf:
        vals = []
        for r in runs:
            if key == "train_time_s":
                vals.append(float(r.get("train_time_s", float("nan"))))
            elif key == "backtest":
                vals.append(float(r["metrics"]["backtest"]["retorno_estrategia"]))
            else:
                vals.append(float(r["metrics"].get(key, float("nan"))))
        ranks = _ranks(vals, hb)
        row = '<tr><td style="padding:6px 10px;border-bottom:1px solid #ccc">%s</td>' % lbl
        for v, rk in zip(vals, ranks):
            if v != v:
                disp = "-"
            elif fmt == "pct":
                disp = "%.1f%%" % (v * 100)
            else:
                disp = "%.1f s" % v
            row += _cell(disp, rk)
        html += row + "</tr>"
    html += "</table>"
    return html


BEFORE_HTML = """
<table style="border-collapse:collapse;width:100%;font-size:13px">
<tr>
<th style="padding:8px 10px;text-align:left;border-bottom:2px solid #1f2d3d">Characteristic</th>
<th style="padding:8px 10px;text-align:center;border-bottom:2px solid #1f2d3d">Light</th>
<th style="padding:8px 10px;text-align:center;border-bottom:2px solid #1f2d3d">Intermediate</th>
<th style="padding:8px 10px;text-align:center;border-bottom:2px solid #1f2d3d">Heavy</th>
</tr>
{rows}
</table>
"""


def before_training_html():
    data = [
        ("Algorithm", "Logistic Regression", "Random Forest", "HistGradientBoosting"),
        ("Type", "Linear model", "Tree ensemble (bagging)", "Tree ensemble (boosting)"),
        ("Relative training speed", "Very fast", "Moderate", "Fast"),
        ("Interpretability", "High", "Medium", "Low"),
        ("Captures non-linear patterns", "Limited", "Yes", "Yes (strong)"),
        ("Overfitting risk", "Low", "Medium", "Medium-high"),
        ("Key hyperparameters", "C, class_weight", "n_estimators, max_depth",
         "learning_rate, max_iter"),
    ]
    rows = ""
    for r in data:
        rows += '<tr><td style="padding:6px 10px;border-bottom:1px solid #ccc;font-weight:600">%s</td>' % r[0]
        for c in r[1:]:
            rows += '<td style="padding:6px 10px;border-bottom:1px solid #ccc;text-align:center">%s</td>' % c
        rows += "</tr>"
    return BEFORE_HTML.format(rows=rows)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("BTC Direction Predictor")
st.sidebar.caption("Bitcoin daily up/down - supervised ML")
st.sidebar.markdown("---")
if data_ready():
    _o = load_ohlcv()
    st.sidebar.metric("Dataset", "%d days" % len(_o))
    st.sidebar.caption("Last day: %s" % _o.index[-1].date())
else:
    st.sidebar.warning("No dataset. Use the Data tab.")
st.sidebar.metric("Trained models", "%d" % len(list_runs(MODELS_DIR)))
if IN_BROWSER:
    st.sidebar.info("Web demo: everything runs in your browser. It starts with the three models "
                    "pre-trained with the default hyperparameters; models you train here are "
                    "lost when the page is reloaded.")

tab_data, tab_train, tab_pred, tab_cmp, tab_hist = st.tabs(
    ["Data", "Training", "Prediction", "Comparison", "History"])


# ----------------------------------------------------------------------
# TAB - DATA
# ----------------------------------------------------------------------
with tab_data:
    st.header("Dataset (Kaggle)")
    st.write("Source: **%s** - minute-level OHLCV data from the Bitstamp exchange." % KAGGLE_DS)
    if data_ready():
        o = load_ohlcv()
        st.success("Processed dataset available: %d days (up to %s)." % (len(o), o.index[-1].date()))
    else:
        st.warning("No processed dataset yet. Download it below.")

    if IN_BROWSER:
        st.info("In the web demo the processed daily dataset is already loaded. Downloading from "
                "Kaggle only works when the app runs locally (`streamlit run app/app.py`).")
    else:
        st.subheader("Download via Kaggle (kagglehub)")
        st.caption("The dataset is public and updated every day, so no account is needed. "
                   "The last, still incomplete day of the file is discarded.")
        with st.expander("Kaggle API token (only if Kaggle asks for authentication)"):
            st.markdown(
                "1. Go to **kaggle.com**, profile picture, **Settings**.\n"
                "2. **API** section, **Create New Token** (format `KGAT_...`).\n"
                "3. Paste the token below. It is used only in memory and never written to disk.")
            token = st.text_input("KAGGLE_API_TOKEN", type="password", placeholder="KGAT_...")
        if st.button("Download latest dataset version", type="primary"):
            try:
                with st.spinner("Downloading from Kaggle and processing (may take 1-2 min)..."):
                    if token.strip():
                        os.environ["KAGGLE_API_TOKEN"] = token.strip()
                    from data_pipeline import download_latest
                    nd, nf = process_raw_to_data(download_latest())
                    st.cache_data.clear()
                st.success("Done: %d days / %d samples." % (nd, nf))
            except ImportError:
                st.error("kagglehub is missing. Install it with: pip install kagglehub")
            except Exception as e:
                st.error("Download failed: %s" % e)

        with st.expander("Reprocess a local file instead (offline)"):
            if st.button("Reprocess local dataset"):
                if os.path.exists(RAW_CSV):
                    try:
                        with st.spinner("Processing local file..."):
                            nd, nf = process_raw_to_data(RAW_CSV)
                            st.cache_data.clear()
                        st.success("Reprocessed: %d days / %d samples." % (nd, nf))
                    except Exception as e:
                        st.error("Failed: %s" % e)
                else:
                    st.error("File not found: %s" % RAW_CSV)

    # ---- Data processing & feature engineering ----
    if data_ready():
        st.markdown("---")
        st.header("Data processing and feature engineering")
        feats = load_feats()
        ohlcv = load_ohlcv()
        pos = float(feats["target"].mean())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Daily candles", "%d" % len(ohlcv))
        k2.metric("Modelling samples", "%d" % len(feats))
        k3.metric("Features", "%d" % len(feature_columns(feats)))
        k4.metric("Target balance (up)", "%.1f%%" % (pos * 100))

        with st.expander("Cleaning steps applied to the raw data"):
            st.markdown(
                "- Resample 1-minute OHLCV into **daily candles** (open=first, high=max, "
                "low=min, close=last, volume=sum).\n"
                "- Remove the **illiquid early period** (before 2015 and days with zero volume).\n"
                "- Fill occasional **calendar gaps** (forward-fill prices, zero volume).\n"
                "- Build the features and drop rows without enough history; the last day has no "
                "next-day label and is removed from the training set.")

        st.subheader("Engineered features by group")
        rows = []
        for g, cols in FEATURE_GROUPS.items():
            for c in cols:
                rows.append({"Group": g, "Feature": c, "Description": FEATURE_DESCRIPTIONS.get(c, "")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Target distribution")
            vc = feats["target"].value_counts().sort_index()
            st.bar_chart(pd.DataFrame({"days": [int(vc.get(0, 0)), int(vc.get(1, 0))]},
                                      index=["DOWN", "UP"]))
        with c2:
            st.subheader("Correlation with target (top 12)")
            corr = feats[feature_columns(feats)].corrwith(feats["target"]).abs().sort_values()
            st.bar_chart(corr.tail(12))

        st.subheader("Feature selection (optional)")
        st.info("Choose which feature groups to use in the next trainings. By default all are used.")
        st.multiselect("Feature groups", list(FEATURE_GROUPS.keys()),
                       default=st.session_state.get("sel_groups", list(FEATURE_GROUPS.keys())),
                       key="sel_groups")
        st.caption("%d features currently selected." % len(selected_cols()))

        st.markdown("---")
        st.header("Add new data to the dataset")
        st.caption("Append new daily rows, then train a new model version in the Training tab.")
        o = load_ohlcv()
        last_close = float(o["Close"].iloc[-1])
        st.write("The dataset currently ends on **%s**." % o.index[-1].date())

        with st.form("add_row_form"):
            st.markdown("**Add a single day**")
            r1 = st.columns(2)
            d = r1[0].date_input("Date", value=(o.index[-1] + pd.Timedelta(days=1)).date())
            vol = r1[1].number_input("Volume (BTC)", min_value=0.0,
                                     value=float(o["Volume"].iloc[-1]), format="%.4f")
            r2 = st.columns(4)
            op = r2[0].number_input("Open", min_value=0.0, value=last_close, format="%.2f")
            hi = r2[1].number_input("High", min_value=0.0, value=last_close, format="%.2f")
            lo = r2[2].number_input("Low", min_value=0.0, value=last_close, format="%.2f")
            cl = r2[3].number_input("Close", min_value=0.0, value=last_close, format="%.2f")
            add_one = st.form_submit_button("Add day to dataset", type="primary")
        if add_one:
            try:
                nd = pd.DataFrame({"Open": [op], "High": [hi], "Low": [lo],
                                   "Close": [cl], "Volume": [vol]},
                                  index=[pd.Timestamp(d, tz="UTC")])
                ndays, nsamp = append_daily_rows(nd)
                st.cache_data.clear()
                st.success("Day added. Dataset now has %d days / %d samples. "
                           "Train a new version in the Training tab." % (ndays, nsamp))
            except Exception as e:
                st.error("Failed to add the row: %s" % e)

        st.markdown("**Or upload a CSV with new daily rows**")
        st.caption("The CSV must have a date column (as index) plus Open, High, Low, Close, Volume.")
        up = st.file_uploader("New daily rows (CSV)", type=["csv"], key="newrows_csv")
        if up is not None and st.button("Append CSV rows"):
            try:
                tmp = pd.read_csv(up, index_col=0, parse_dates=True)
                ndays, nsamp = append_daily_rows(tmp)
                st.cache_data.clear()
                st.success("Rows appended. Dataset now has %d days / %d samples. "
                           "Train a new version in the Training tab." % (ndays, nsamp))
            except Exception as e:
                st.error("Failed to append the CSV: %s" % e)


# ----------------------------------------------------------------------
# TAB - TRAINING
# ----------------------------------------------------------------------
with tab_train:
    st.header("Training")
    if not data_ready():
        st.warning("No dataset available. Use the Data tab to download it.")
    else:
        feats = load_feats()
        job = st.session_state.get("job")
        if job is not None:
            if job["shared"].get("status") == "running" and job["thread"].is_alive():
                _render_running(job)
                time.sleep(0.4)
                st.rerun()
            else:
                _consume_job(job, feats)
                st.rerun()
        else:
            _render_train_controls(feats)
            _render_last_train()


# ----------------------------------------------------------------------
# TAB - PREDICTION
# ----------------------------------------------------------------------
with tab_pred:
    st.header("Prediction")
    runs = list_runs(MODELS_DIR)
    reg = load_registry(MODELS_DIR)
    if not data_ready():
        st.warning("No dataset available.")
    elif not runs:
        st.warning("No trained models yet. Train one in the Training tab.")
    else:
        opt = {r["id"]: r for r in runs}

        def run_label(rid):
            r = opt[rid]
            return "%s - %s - AUC %.3f [%s]" % (MODEL_REGISTRY[r["family"]]["label"],
                                                r["trained_at"], r["metrics"]["roc_auc"], rid)

        sel = st.selectbox("Trained model", list(opt.keys()), format_func=run_label)
        ohlcv = load_ohlcv()
        feats_pred = build_features(ohlcv, with_target=False)
        run = opt[sel]
        cols = run.get("feature_cols") or reg.get("feature_cols") or feature_columns(build_features(ohlcv))

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Last available day", str(ohlcv.index[-1].date()))
            st.metric("Last close (USD)", format(float(ohlcv["Close"].iloc[-1]), ",.2f"))
        with c2:
            base_mode = st.radio("Base day", ["Last available day", "Pick a historical date"])
            if base_mode == "Last available day":
                base_dt = feats_pred.index[-1]
            else:
                d = st.date_input("Base date", value=feats_pred.index[-2].date(),
                                  min_value=feats_pred.index[0].date(),
                                  max_value=feats_pred.index[-1].date())
                base_dt = pd.Timestamp(d, tz="UTC")
                if base_dt not in feats_pred.index:
                    p = feats_pred.index.get_indexer([base_dt], method="ffill")[0]
                    base_dt = feats_pred.index[p]

        if st.button("Predict next day direction", type="primary"):
            model = load_run(MODELS_DIR, sel)
            x = feats_pred.loc[[base_dt], cols].astype("float64")
            proba_up = float(model.predict_proba(x)[0, 1])
            pred = int(proba_up >= 0.5)
            pred_date = (base_dt + pd.Timedelta(days=1)).date()
            r1, r2, r3 = st.columns(3)
            r1.metric("Base day", str(base_dt.date()))
            r2.metric("Prediction for %s" % pred_date, LABELS[pred])
            r3.metric("Probability of going up", "%.1f%%" % (proba_up * 100))
            st.progress(proba_up)
            append_history({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "model": run_label(sel), "base_date": str(base_dt.date()),
                            "predicted_date": str(pred_date), "prob_up": round(proba_up, 4),
                            "prediction": LABELS[pred]})
            st.success("Prediction saved to history.")

        st.markdown("---")
        st.subheader("Price (last 180 days)")
        st.line_chart(ohlcv["Close"].iloc[-180:])


# ----------------------------------------------------------------------
# TAB - COMPARISON
# ----------------------------------------------------------------------
with tab_cmp:
    st.header("Model comparison")

    st.subheader("Before training - model characteristics")
    st.caption("Qualitative comparison of the three model families, independent of any training.")
    st.markdown(before_training_html(), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("After training - your trained models")
    runs = list_runs(MODELS_DIR)
    if not runs:
        st.info("No trained models yet. Train models in the Training tab to populate this table.")
    else:
        best = max(runs, key=lambda r: (round(r["metrics"]["roc_auc"], 4), round(r["metrics"]["f1"], 4)))
        worst = min(runs, key=lambda r: (round(r["metrics"]["roc_auc"], 4), round(r["metrics"]["f1"], 4)))
        b1, b2 = st.columns(2)
        b1.success("Best model: %s [%s] (ROC-AUC %.3f, F1 %.3f)"
                   % (MODEL_REGISTRY[best["family"]]["label"], best["id"],
                      best["metrics"]["roc_auc"], best["metrics"]["f1"]))
        if worst["id"] != best["id"]:
            b2.error("Weakest model: %s [%s] (ROC-AUC %.3f, F1 %.3f)"
                     % (MODEL_REGISTRY[worst["family"]]["label"], worst["id"],
                        worst["metrics"]["roc_auc"], worst["metrics"]["f1"]))
        st.markdown(after_training_html(runs), unsafe_allow_html=True)
        st.caption("Columns are the trained model versions; rows are the hyperparameters used and "
                   "the resulting metrics. Green up = best, red down = weakest across the row.")

        st.markdown("#### Per-model details")
        for r in runs:
            with st.expander("%s [%s] - trained at %s"
                             % (MODEL_REGISTRY[r["family"]]["label"], r["id"], r.get("trained_at", "-"))):
                m = r["metrics"]
                cm = np.array(m["confusion_matrix"])
                cmdf = pd.DataFrame(cm, index=["Real DOWN", "Real UP"],
                                    columns=["Pred DOWN", "Pred UP"])
                d1, d2 = st.columns(2)
                d1.write("**Confusion matrix**"); d1.dataframe(cmdf, width="stretch")
                bt = m["backtest"]
                d2.metric("Backtest strategy", "%.1f%%" % (bt["retorno_estrategia"] * 100),
                          delta="%.1f pp vs buy & hold"
                          % ((bt["retorno_estrategia"] - bt["retorno_buy_and_hold"]) * 100))
                d2.metric("Training time", "%.1f s" % r.get("train_time_s", 0.0))
                if r.get("lc"):
                    st.pyplot(learning_curve_fig(r["lc"]))


# ----------------------------------------------------------------------
# TAB - HISTORY
# ----------------------------------------------------------------------
with tab_hist:
    st.header("Prediction history")
    hist = read_history()
    if hist.empty:
        st.info("No predictions yet. Use the Prediction tab.")
    else:
        ohlcv = load_ohlcv() if data_ready() else None

        def outcome(row):
            try:
                if ohlcv is None:
                    return "?"
                d0 = pd.Timestamp(row["base_date"], tz="UTC")
                d1 = pd.Timestamp(row["predicted_date"], tz="UTC")
                if d0 in ohlcv.index and d1 in ohlcv.index:
                    real_up = ohlcv.loc[d1, "Close"] > ohlcv.loc[d0, "Close"]
                    return "correct" if real_up == (row["prediction"] == "UP") else "wrong"
            except Exception:
                pass
            return "pending"

        hist["outcome"] = hist.apply(outcome, axis=1)
        st.dataframe(hist.iloc[::-1].reset_index(drop=True), width="stretch")
        conf = hist[hist["outcome"].isin(["correct", "wrong"])]
        if len(conf):
            acc = (conf["outcome"] == "correct").mean()
            st.metric("Hit rate (confirmed predictions)", "%.1f%% (%d)" % (acc * 100, len(conf)))
        if st.button("Clear history"):
            try:
                open(HISTORY_CSV, "w").close()
            except Exception:
                pass
            st.rerun()
