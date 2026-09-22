"""
Pipeline de dados (linha de comandos)
=====================================
Bitcoin (BTC/USD) - previsão da direção diária (sobe/desce).

  1. Carrega os dados OHLCV ao minuto (~7,6 milhões de linhas, 2012-2026)
  2. Reamostra para velas diárias, remove o período ilíquido inicial e trata falhas
  3. Constrói as features técnicas e o alvo (fecho de amanhã > fecho de hoje)
  4. Grava data/btc_daily_ohlcv.csv e data/btc_daily_features.csv

A lógica de transformação vive em features.py e é a mesma usada pela aplicação,
para que os dados de treino e os de previsão sejam construídos exatamente da mesma forma.

Uso:
    python src/data_pipeline.py [caminho/para/btcusd_1-min_data.csv]
"""
import os
import sys

import pandas as pd

from features import build_features, feature_columns, resample_daily

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("OUT_DIR", os.path.join(BASE, "data"))
DEFAULT_RAW_CSV = os.environ.get("RAW_CSV", os.path.join(DATA_DIR, "btcusd_1-min_data.csv"))


def load_minute_data(path: str) -> pd.DataFrame:
    """Lê o CSV ao minuto do Kaggle e indexa-o por data/hora UTC."""
    df = pd.read_csv(
        path,
        dtype={"Timestamp": "int64", "Open": "float32", "High": "float32",
               "Low": "float32", "Close": "float32", "Volume": "float32"},
    )
    df["datetime"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    return df.set_index("datetime").sort_index()


def main(raw_csv: str = DEFAULT_RAW_CSV) -> None:
    if not os.path.exists(raw_csv):
        sys.exit(f"Ficheiro não encontrado: {raw_csv}\n"
                 "Descarregue-o do Kaggle (mczielinski/bitcoin-historical-data) ou use a aplicação.")

    print(f"[1] A carregar {raw_csv} ...")
    minute = load_minute_data(raw_csv)
    print(f"    {len(minute):,} linhas | {minute.index.min()} -> {minute.index.max()}")

    print("[2] A reamostrar para velas diárias ...")
    daily = resample_daily(minute)
    print(f"    {len(daily):,} dias")

    print("[3] A construir features ...")
    feats = build_features(daily, with_target=True)
    cols = feature_columns(feats)
    print(f"    {len(feats):,} amostras | {len(cols)} features")
    print(f"    Alvo: {feats['target'].mean():.1%} dias 'sobe' | {1 - feats['target'].mean():.1%} 'desce'")

    os.makedirs(DATA_DIR, exist_ok=True)
    feats.to_csv(os.path.join(DATA_DIR, "btc_daily_features.csv"))
    daily.to_csv(os.path.join(DATA_DIR, "btc_daily_ohlcv.csv"))
    print(f"[OK] Dados gravados em {DATA_DIR}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW_CSV)
