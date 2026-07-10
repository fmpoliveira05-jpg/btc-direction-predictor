"""
Pipeline de dados e Feature Engineering
=======================================
Bitcoin (BTC/USD) - Previsão de direção diária (sobe/desce)

Trabalho Prático de Inteligência Artificial - ESTG P.PORTO
Francisco Miguel Pereira Oliveira (8230148)

Este script:
  1. Carrega os dados OHLCV ao minuto (~7.6M linhas, 2012-2026)
  2. Remove o período ilíquido inicial (volume ~0)
  3. Reamostra de 1-min -> velas DIÁRIAS (OHLCV)
  4. Trata gaps temporais
  5. Constrói features técnicas (sem look-ahead bias)
  6. Define o target binário: fecho de AMANHÃ > fecho de HOJE ?
  7. Grava o dataset processado para treino dos modelos.
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------
RAW_CSV = os.environ.get(
    "RAW_CSV",
    "/sessions/festive-modest-planck/mnt/uploads/btcusd_1-min_data.csv",
)
OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/sessions/festive-modest-planck/mnt/TP - ER (Melhoria)/projeto/data",
)
os.makedirs(OUT_DIR, exist_ok=True)

# Período mínimo de liquidez. Antes disto o BTC era pouco transacionado
# (preços "congelados" e volume ~0), o que polui o sinal.
START_DATE = "2015-01-01"


# ----------------------------------------------------------------------
# 1. Carregar dados ao minuto
# ----------------------------------------------------------------------
def load_minute_data(path: str) -> pd.DataFrame:
    print(f"[1] A carregar {path} ...")
    df = pd.read_csv(
        path,
        dtype={
            "Timestamp": "int64",
            "Open": "float32",
            "High": "float32",
            "Low": "float32",
            "Close": "float32",
            "Volume": "float32",
        },
    )
    df["datetime"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    df = df.set_index("datetime").sort_index()
    print(f"    {len(df):,} linhas | {df.index.min()} -> {df.index.max()}")
    return df


# ----------------------------------------------------------------------
# 2/3. Reamostrar para velas diárias
# ----------------------------------------------------------------------
def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    print("[2] A reamostrar 1-min -> diário (OHLCV) ...")
    daily = df.resample("1D").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    # Remover dias sem qualquer transação (volume 0) -> período ilíquido / gaps
    daily = daily[daily["Volume"] > 0]
    daily = daily[daily.index >= pd.Timestamp(START_DATE, tz="UTC")]
    # Dias soltos em falta: reindexar para calendário contínuo e preencher
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="1D", tz="UTC")
    n_missing = len(full_idx) - len(daily)
    daily = daily.reindex(full_idx)
    # Preço: forward-fill (último preço conhecido); Volume em falta -> 0
    daily[["Open", "High", "Low", "Close"]] = daily[
        ["Open", "High", "Low", "Close"]
    ].ffill()
    daily["Volume"] = daily["Volume"].fillna(0.0)
    daily = daily.dropna()
    print(f"    {len(daily):,} dias | gaps preenchidos: {n_missing}")
    return daily


# ----------------------------------------------------------------------
# 4. Indicadores técnicos (apenas informação passada -> sem leakage)
# ----------------------------------------------------------------------
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    print("[3] A construir features técnicas ...")
    d = daily.copy()
    close = d["Close"]

    # --- Retornos ---
    d["return_1d"] = close.pct_change(1)
    for lag in range(1, 6):  # retornos desfasados (memória curta)
        d[f"return_lag{lag}"] = d["return_1d"].shift(lag)
    d["return_5d"] = close.pct_change(5)
    d["return_10d"] = close.pct_change(10)

    # --- Médias móveis e rácios preço/MA ---
    for w in (5, 10, 20, 50):
        ma = close.rolling(w).mean()
        d[f"ma{w}_ratio"] = close / ma - 1.0
    d["ma_cross_5_20"] = (close.rolling(5).mean() / close.rolling(20).mean()) - 1.0

    # --- Volatilidade (desvio-padrão dos retornos) ---
    for w in (5, 10, 20):
        d[f"vol_{w}d"] = d["return_1d"].rolling(w).std()

    # --- RSI ---
    d["rsi_14"] = rsi(close, 14)

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd"] = macd / close          # normalizado pelo preço
    d["macd_hist"] = (macd - signal) / close

    # --- Bandas de Bollinger (posição relativa) ---
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    d["bb_position"] = (close - ma20) / (2 * std20)

    # --- Amplitude diária (range) ---
    d["hl_range"] = (d["High"] - d["Low"]) / close
    d["co_change"] = (d["Close"] - d["Open"]) / d["Open"]

    # --- Volume ---
    d["vol_change"] = d["Volume"].pct_change(1)
    d["vol_ma_ratio"] = d["Volume"] / d["Volume"].rolling(20).mean() - 1.0

    # --- Sazonalidade ---
    d["dayofweek"] = d.index.dayofweek

    # --- TARGET: fecho de AMANHÃ > fecho de HOJE ? ---
    d["target"] = (close.shift(-1) > close).astype("int8")

    # A última linha não tem target (não há "amanhã") -> remover
    d = d.iloc[:-1]

    # Limpar infinitos e NaN introduzidos pelas janelas
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"    {len(d):,} amostras finais | {d.shape[1]} colunas")
    return d


def main():
    df = load_minute_data(RAW_CSV)
    daily = resample_daily(df)
    feats = build_features(daily)

    feature_cols = [c for c in feats.columns if c not in
                    ("Open", "High", "Low", "Close", "Volume", "target")]

    # Distribuição do target
    pos = feats["target"].mean()
    print(f"[4] Target: {pos:.1%} dias 'sobe' | {1-pos:.1%} dias 'desce'")
    print(f"    Features ({len(feature_cols)}): {feature_cols}")

    out_path = os.path.join(OUT_DIR, "btc_daily_features.csv")
    feats.to_csv(out_path)
    print(f"[OK] Gravado: {out_path}")

    # Também guardar OHLCV diário "limpo" (útil para a app / gráficos)
    daily.to_csv(os.path.join(OUT_DIR, "btc_daily_ohlcv.csv"))
    print(f"[OK] Gravado: {os.path.join(OUT_DIR, 'btc_daily_ohlcv.csv')}")


if __name__ == "__main__":
    main()
