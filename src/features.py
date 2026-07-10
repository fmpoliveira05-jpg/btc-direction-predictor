"""
Feature engineering (modulo reutilizavel)
=========================================
Usado tanto pelo pipeline inicial como pela app (re-treino com novos dados),
garantindo que as features sao construidas EXATAMENTE da mesma forma.

Todas as features usam apenas informacao passada -> sem look-ahead bias.
"""
import numpy as np
import pandas as pd

START_DATE = "2015-01-01"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def resample_daily(minute_df: pd.DataFrame) -> pd.DataFrame:
    """1-min OHLCV -> velas diarias. Espera indice datetime (UTC)."""
    daily = minute_df.resample("1D").agg(
        Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
        Close=("Close", "last"), Volume=("Volume", "sum"))
    daily = daily[daily["Volume"] > 0]
    daily = daily[daily.index >= pd.Timestamp(START_DATE, tz="UTC")]
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="1D", tz="UTC")
    daily = daily.reindex(full)
    daily[["Open", "High", "Low", "Close"]] = daily[["Open", "High", "Low", "Close"]].ffill()
    daily["Volume"] = daily["Volume"].fillna(0.0)
    return daily.dropna()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(daily: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    """Recebe OHLCV diario e devolve DataFrame com features (+ target opcional)."""
    d = daily.copy()
    close = d["Close"]

    d["return_1d"] = close.pct_change(1)
    for lag in range(1, 6):
        d["return_lag%d" % lag] = d["return_1d"].shift(lag)
    d["return_5d"] = close.pct_change(5)
    d["return_10d"] = close.pct_change(10)

    for w in (5, 10, 20, 50):
        d["ma%d_ratio" % w] = close / close.rolling(w).mean() - 1.0
    d["ma_cross_5_20"] = (close.rolling(5).mean() / close.rolling(20).mean()) - 1.0

    for w in (5, 10, 20):
        d["vol_%dd" % w] = d["return_1d"].rolling(w).std()

    d["rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd"] = macd / close
    d["macd_hist"] = (macd - signal) / close

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    d["bb_position"] = (close - ma20) / (2 * std20)

    d["hl_range"] = (d["High"] - d["Low"]) / close
    d["co_change"] = (d["Close"] - d["Open"]) / d["Open"]
    d["vol_change"] = d["Volume"].pct_change(1)
    d["vol_ma_ratio"] = d["Volume"] / d["Volume"].rolling(20).mean() - 1.0
    d["dayofweek"] = d.index.dayofweek

    if with_target:
        d["target"] = (close.shift(-1) > close).astype("int8")
        d = d.iloc[:-1]  # ultima linha nao tem "amanha"

    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    return d


def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in OHLCV + ["target"]]


# Agrupamento das features (usado na interface para selecao opcional)
FEATURE_GROUPS = {
    "Returns": ["return_1d", "return_lag1", "return_lag2", "return_lag3",
                "return_lag4", "return_lag5", "return_5d", "return_10d"],
    "Moving averages": ["ma5_ratio", "ma10_ratio", "ma20_ratio", "ma50_ratio", "ma_cross_5_20"],
    "Volatility": ["vol_5d", "vol_10d", "vol_20d"],
    "Technical indicators": ["rsi_14", "macd", "macd_hist", "bb_position"],
    "Range & volume": ["hl_range", "co_change", "vol_change", "vol_ma_ratio"],
    "Seasonality": ["dayofweek"],
}

# Descricao curta de cada feature (para a interface / relatorio)
FEATURE_DESCRIPTIONS = {
    "return_1d": "Daily return (close-to-close).",
    "return_lag1": "Return lagged by 1 day.",
    "return_lag2": "Return lagged by 2 days.",
    "return_lag3": "Return lagged by 3 days.",
    "return_lag4": "Return lagged by 4 days.",
    "return_lag5": "Return lagged by 5 days.",
    "return_5d": "Cumulative 5-day return.",
    "return_10d": "Cumulative 10-day return.",
    "ma5_ratio": "Price vs 5-day moving average.",
    "ma10_ratio": "Price vs 10-day moving average.",
    "ma20_ratio": "Price vs 20-day moving average.",
    "ma50_ratio": "Price vs 50-day moving average.",
    "ma_cross_5_20": "5-day vs 20-day moving-average crossover.",
    "vol_5d": "5-day return volatility (std).",
    "vol_10d": "10-day return volatility (std).",
    "vol_20d": "20-day return volatility (std).",
    "rsi_14": "Relative Strength Index (14).",
    "macd": "MACD line (normalised by price).",
    "macd_hist": "MACD histogram (MACD minus signal).",
    "bb_position": "Position within the Bollinger Bands.",
    "hl_range": "Daily high-low range.",
    "co_change": "Close vs open change.",
    "vol_change": "Daily volume change.",
    "vol_ma_ratio": "Volume vs its 20-day average.",
    "dayofweek": "Day of the week (seasonality).",
}
