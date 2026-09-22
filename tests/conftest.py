import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


@pytest.fixture
def daily():
    """300 dias de preços sintéticos (passeio aleatório reprodutível)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=300, freq="1D", tz="UTC")
    close = 10_000 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx))))
    open_ = close * (1 + rng.normal(0, 0.005, len(idx)))
    return pd.DataFrame({
        "Open": open_,
        "High": np.maximum(open_, close) * 1.01,
        "Low": np.minimum(open_, close) * 0.99,
        "Close": close,
        "Volume": rng.uniform(100, 1_000, len(idx)),
    }, index=idx)
