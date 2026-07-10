# BTC Direction Predictor - Trabalho Pratico de Inteligencia Artificial

**Francisco Miguel Pereira Oliveira (8230148)** - ESTG P.PORTO - 2025/2026 (Epoca de Recurso)

Previsao da **direcao diaria** do preco do Bitcoin (sobe/desce) - classificacao binaria
supervisionada, com 3 modelos scikit-learn e uma aplicacao Streamlit interativa.

## Estrutura

```
projeto/
  data/
    btc_daily_ohlcv.csv        # OHLCV diario (reamostrado do 1-min)
    btc_daily_features.csv     # dataset com 25 features + target
  src/
    features.py                # feature engineering + grupos de features (reutilizavel)
    modeling.py                # treino / AutoML / learning curve / versoes (reutilizavel)
    data_pipeline.py           # 1-min -> diario -> features (linha de comandos)
    train_models.py            # treino completo com GridSearchCV + baselines (linha de comandos)
  models/
    registry.json              # registo das versoes treinadas pelo utilizador (inicia vazio)
    run_XXXX.joblib            # cada treino feito na app gera uma versao (nao versionar)
    metrics.json               # metricas do treino via train_models.py (referencia do relatorio)
  app/
    app.py                     # aplicacao Streamlit
```

> A app **nao** traz modelos pre-treinados: cada modelo e treinado pelo utilizador na propria
> aplicacao (evita incompatibilidades de versao do scikit-learn e segue a logica "treina e so
> depois preve"). Por isso o `registry.json` inicia vazio e nao ha ficheiros `run_*.joblib`.

> O dataset bruto ao minuto (`btcusd_1-min_data.csv`, ~386 MB, 7.6M linhas) nao esta incluido por
> dimensao; obtem-se em Kaggle (mczielinski/bitcoin-historical-data). Pode ser descarregado
> diretamente pela app (separador Data) ou colocado em `data/` e reprocessado. Os CSV diarios ja
> processados estao incluidos.

## Como executar

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Depois, na aplicacao (toda em ingles), segue os separadores por esta ordem:

1. **Data** - descarrega a ultima versao do dataset do Kaggle (token num campo protegido, nunca
   gravado) ou reprocessa um CSV local; mostra o processamento/feature engineering, permite
   **selecionar grupos de features** e **adicionar novas linhas de dados** ao dataset.
2. **Training** - escolhe um modelo (Light / Intermediate / Heavy) e ajusta os hiperparametros, ou
   usa o **AutoML**. Mostra a learning curve, um cronometro e um botao de Stop. Cada treino fica
   guardado como uma **nova versao**.
3. **Prediction** - escolhe a versao treinada e obtem a direcao prevista e a probabilidade de subida.
4. **Comparison** - compara os modelos antes do treino (caracteristicas) e depois do treino
   (hiperparametros + metricas por modelo, com indicadores de melhor/pior).
5. **History** - historico das previsoes feitas, com verificacao de acerto e taxa de acerto.

> Alternativa por linha de comandos (opcional): `python src/data_pipeline.py` (gera os CSV diarios)
> e `python src/train_models.py` (treina os 3 modelos com GridSearchCV e grava `metrics.json`).

## Funcionalidades da aplicacao

**Objetivo principal (obrigatorio):** obter previsoes interativas a partir de um modelo treinado
(separador Prediction).

**Bonificacao (todas implementadas):**
- Selecao de um modelo, de entre varios disponiveis (separador **Prediction**).
- Visualizacao das metricas de qualidade de cada modelo (separador **Comparison**).
- Historico das previsoes feitas, com verificacao de acerto (separador **History**).
- Adicao de novas linhas de dados ao dataset (separador **Data**) e treino de uma nova versao do
  modelo (separador **Training**).

**Extras:** download integrado do Kaggle, AutoML, selecao de grupos de features, learning curve,
cronometro e interrupcao de treino.
