# energie-app

## Objectif
Prédire les **charges thermiques (chauffage et climatisation)** d'un bâtiment à partir
de ses caractéristiques enveloppe, de l'usage et de variables météo, avec un front
Streamlit interactif. Horizon : prédiction à J+1 à partir des valeurs des 30 derniers jours.

## Données
Dataset **UCI Energy Efficiency (Tsanas & Xifara, 2012)** enrichi avec des données
climatiques quotidiennes (fichier source : `ENB2012_with_original_weather27.xlsx`).

Features :
- **Enveloppe (ENB2012)** : `Relative_Compactness`, `Surface_Area`, `Wall_Area`,
  `Roof_Area`, `Overall_Height`, `Orientation`, `Glazing_Area`, `Glazing_Area_Distribution`
- **Météo** : `temp`, `humidity`, `irradiation`, `wind_speed`
- **Usage (encodé)** : `usage_type_encoded` ∈ {Résidentiel, Bureau, Éducatif,
  Commercial, Industriel}
- **Temporelles** : `month`, `Heating_Load_lag_1/2/3/7`, `Cooling_Load_lag_1/2/3/7`
- **Cibles** : `Heating_Load` (kW), `Cooling_Load` (kW)

## Méthode
Comparaison de 5 approches sur les 30 derniers jours comme test (hold-out temporel) :

1. **LightGBM** + lags + exogènes
2. **XGBoost** + MinMaxScaler
3. **ARIMA + LSTM** (hybride : ARIMA sur la tendance + LSTM sur les résidus)
4. **LSTM Multivarié** (séquences de 30 jours)
5. **SARIMA** et **LSTM univarié** comme références

Sélection du modèle via sidebar Streamlit.

## Stack
- Python, pandas, numpy
- **LightGBM**, **XGBoost**, scikit-learn
- **TensorFlow / Keras** (LSTM)
- **statsmodels** (ARIMA / SARIMA)
- **Streamlit** (UI)
- `joblib` pour la persistance des modèles

## Résultats
Évaluation sur les 30 derniers jours réels :

| Modèle | Variable | MAE (kW) | RMSE (kW) | MAPE | R² |
|---|---|---|---|---|---|
| **XGBoost** ⭐ | Heating | 0,47 | 0,96 | 2,28 % | **0,99** |
| **XGBoost** ⭐ | Cooling | 0,79 | 1,24 | 2,99 % | **0,98** |
| **LightGBM** | Heating | 0,55 | 0,75 | 1,62 % | 0,935 |
| **LightGBM** | Cooling | 0,73 | 0,99 | 4,39 % | 0,886 |
| LSTM Multivarié | Heating | 3,43 | 4,19 | 11,34 % | 0,123 |
| LSTM Multivarié | Cooling | 4,46 | 5,71 | 22,00 % | -0,487 |
| SARIMA | Heating | 2,01 | 3,08 | 6,36 % | - |
| SARIMA | Cooling | 2,68 | 3,11 | 15,93 % | - |
| LSTM univarié | Heating | 2,91 | 3,28 | 8,49 % | - |
| LSTM univarié | Cooling | 1,91 | 2,95 | 10,27 % | - |
| Naïf (baseline) | Heating | 2,14 | 2,93 | 6,57 % | - |
| Naïf (baseline) | Cooling | 2,50 | 3,02 | 14,67 % | - |

**XGBoost et LightGBM** dominent largement les approches séquentielles sur ce dataset
(l'enveloppe et la météo expliquent l'essentiel de la variance, les lags apportent le reste).

## Comment lancer
```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Les modèles entraînés sont fournis dans `models/` (voir cette liste dans le dossier) :
`lgbm_heating_model.pkl`, `lgbm_cooling_model.pkl`, `xgboost_Heating_Load.pkl`,
`xgboost_Cooling_Load.pkl`, `arima_Heating_Load.pkl`, `arima_Cooling_Load.pkl`,
`lstm_Heating_Load.h5`, `lstm_Cooling_Load.h5`, `lstm_multi_heat_model.h5`,
`lstm_multi_cool_model.h5` + leurs scalers et configs.

Démarrer l'app complète (4 modèles) :
```bash
streamlit run app.py
```

Ou la version simplifiée (LightGBM + XGBoost uniquement) :
```bash
streamlit run app_simple.py
```
