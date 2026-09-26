#!/usr/bin/env bash
# Junta numa pasta os ficheiros da demonstração web (GitHub Pages).
# Uso: demo/montar-site.sh _site
set -euo pipefail
DEST="${1:-_site}"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
rm -rf "$DEST"
mkdir -p "$DEST/app" "$DEST/src" "$DEST/data" "$DEST/modelos"
cp "$RAIZ/demo/index.html" "$DEST/index.html"
cp "$RAIZ/app/app.py" "$DEST/app/"
cp "$RAIZ/src/features.py" "$RAIZ/src/modeling.py" "$DEST/src/"
cp "$RAIZ/data/btc_daily_ohlcv.csv" "$RAIZ/data/btc_daily_features.csv" "$DEST/data/"
cp "$RAIZ/demo/modelos/"* "$DEST/modelos/"
touch "$DEST/.nojekyll"
echo "Site montado em $DEST"
