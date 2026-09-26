"""A demonstração web (demo/index.html) monta estes ficheiros: têm de existir e de ser coerentes."""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")


def test_modelos_pre_treinados_existem_e_batem_com_o_registo():
    with open(os.path.join(DEMO, "modelos", "registry.json")) as f:
        reg = json.load(f)
    familias = [r["family"] for r in reg["runs"]]
    assert familias == ["logistic_regression", "random_forest", "hist_gradient_boosting"]
    for r in reg["runs"]:
        assert os.path.getsize(os.path.join(DEMO, "modelos", r["id"] + ".joblib")) > 0
        assert r["metrics"]["cv_roc_auc"] == r["metrics"]["cv_roc_auc"]  # não é NaN


def test_pagina_monta_ficheiros_que_existem():
    with open(os.path.join(DEMO, "index.html")) as f:
        html = f.read()
    lista = re.search(r"const ficheiros = \[(.*?)\];", html, re.S).group(1)
    for caminho in re.findall(r'"([^"]+)"', lista):
        assert os.path.exists(os.path.join(ROOT, caminho)), caminho
    for nome in re.findall(r'"(registry\.json|run_\d+\.joblib)"', html):
        assert os.path.exists(os.path.join(DEMO, "modelos", nome)), nome
