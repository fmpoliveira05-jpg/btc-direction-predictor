"""Teste de fumo: a aplicação Streamlit arranca e mostra os cinco separadores sem erros."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "app.py")


def test_aplicacao_arranca_sem_excecoes():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs][:5] == ["Data", "Training", "Prediction", "Comparison", "History"]
