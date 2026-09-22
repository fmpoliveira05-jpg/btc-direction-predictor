import numpy as np
import pandas as pd

from features import FEATURE_GROUPS, build_features, feature_columns, resample_daily


def test_alvo_indica_se_o_fecho_de_amanha_e_superior(daily):
    feats = build_features(daily)
    close = daily["Close"]
    expected = (close.shift(-1) > close).astype(int).loc[feats.index]
    assert (feats["target"] == expected).all()
    assert feats.index[-1] < daily.index[-1], "o último dia não tem 'amanhã' e é removido"


def test_features_nao_usam_informacao_do_futuro(daily):
    """Alterar os preços a partir de um dia não pode mudar as features dos dias anteriores."""
    cut = daily.index[200]
    before = build_features(daily, with_target=False)
    changed = daily.copy()
    changed.loc[changed.index >= cut, ["Open", "High", "Low", "Close"]] *= 3
    changed.loc[changed.index >= cut, "Volume"] *= 10
    after = build_features(changed, with_target=False)

    cols = feature_columns(before)
    past = before.index[before.index < cut]
    pd.testing.assert_frame_equal(before.loc[past, cols], after.loc[past, cols])


def test_sem_valores_infinitos_ou_em_falta(daily):
    feats = build_features(daily)
    values = feats[feature_columns(feats)].to_numpy(dtype=float)
    assert np.isfinite(values).all()


def test_grupos_de_features_cobrem_todas_as_colunas(daily):
    feats = build_features(daily)
    grouped = {c for cols in FEATURE_GROUPS.values() for c in cols}
    assert grouped == set(feature_columns(feats))


def test_reamostragem_para_velas_diarias():
    idx = pd.date_range("2015-03-01", periods=3 * 1440, freq="1min", tz="UTC")
    minute = pd.DataFrame({
        "Open": np.arange(len(idx), dtype=float),
        "High": np.arange(len(idx), dtype=float) + 1,
        "Low": np.arange(len(idx), dtype=float) - 1,
        "Close": np.arange(len(idx), dtype=float) + 0.5,
        "Volume": np.ones(len(idx)),
    }, index=idx)
    # Um dia inteiro sem transações no meio.
    minute.loc["2015-03-02", "Volume"] = 0

    daily = resample_daily(minute)

    assert len(daily) == 3
    first = daily.iloc[0]
    assert first["Open"] == 0 and first["Close"] == 1439.5 and first["Volume"] == 1440
    # O dia sem volume é preenchido com o preço do dia anterior e volume 0.
    assert daily.iloc[1]["Close"] == first["Close"]
    assert daily.iloc[1]["Volume"] == 0


def test_periodo_iliquido_anterior_a_2015_e_removido():
    idx = pd.date_range("2014-12-30", periods=4 * 1440, freq="1min", tz="UTC")
    minute = pd.DataFrame({c: 1.0 for c in ["Open", "High", "Low", "Close", "Volume"]}, index=idx)
    daily = resample_daily(minute)
    assert daily.index.min() == pd.Timestamp("2015-01-01", tz="UTC")
