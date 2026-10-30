import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# --- Charger les modèles ---
model_lgbm_heat = joblib.load('models/lgbm_heating_model_simple.pkl')
model_lgbm_cool = joblib.load('models/lgbm_cooling_model_simple.pkl')
model_xgb_heat = joblib.load('models/xgb_heating_load_model.pkl')
model_xgb_cool = joblib.load('models/xgb_cooling_load_model.pkl')

# --- Features pour LightGBM et XGBoost ---
features_lgbm = [
    'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
    'Overall_Height', 'temp', 'humidity', 'irradiation', 'wind_speed',
    'month', 'Heating_Load_lag1', 'Cooling_Load_lag1',
    'usage_type_Bureau', 'usage_type_Commercial', 'usage_type_Industriel',
    'usage_type_Résidentiel', 'usage_type_Éducatif',
    'Heating_Load_lag_1', 'Heating_Load_lag_2', 'Heating_Load_lag_3', 'Heating_Load_lag_7'
]

features_xgb = [
    'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
    'Overall_Height', 'temp', 'humidity', 'irradiation', 'wind_speed',
    'month', 'Heating_Load_lag1', 'Cooling_Load_lag1',
    'usage_type_Bureau', 'usage_type_Commercial', 'usage_type_Industriel',
    'usage_type_Résidentiel', 'usage_type_Éducatif'
]

st.title("Prédiction des charges thermiques – Heating & Cooling Load")

# --- Choix du modèle ---
model_choice = st.sidebar.selectbox(
    "Choisir le modèle",
    ["LightGBM", "XGBoost"]
)

# --- Inputs utilisateur ---
st.sidebar.header("Entrer les valeurs des paramètres")
input_data = {
    'Relative_Compactness': st.sidebar.number_input('Relative Compactness', value=0.75),
    'Surface_Area': st.sidebar.number_input('Surface Area', value=500.0),
    'Wall_Area': st.sidebar.number_input('Wall Area', value=300.0),
    'Roof_Area': st.sidebar.number_input('Roof Area', value=150.0),
    'Overall_Height': st.sidebar.number_input('Overall Height', value=7.0),
    'temp': st.sidebar.number_input('Température', value=25.0),
    'humidity': st.sidebar.number_input('Humidité', value=50.0),
    'irradiation': st.sidebar.number_input('Irradiation', value=300.0),
    'wind_speed': st.sidebar.number_input('Vitesse du vent', value=5.0),
    'month': st.sidebar.number_input('Mois (1-12)', value=1),
    'Heating_Load_lag1': st.sidebar.number_input('Heating_Load lag1', value=100.0),
    'Cooling_Load_lag1': st.sidebar.number_input('Cooling_Load lag1', value=50.0),
    'usage_type_Bureau': st.sidebar.number_input('Bureau', value=0),
    'usage_type_Commercial': st.sidebar.number_input('Commercial', value=0),
    'usage_type_Industriel': st.sidebar.number_input('Industriel', value=0),
    'usage_type_Résidentiel': st.sidebar.number_input('Résidentiel', value=1),
    'usage_type_Éducatif': st.sidebar.number_input('Éducatif', value=0),
}

# --- Création DataFrame pour prédiction ---
input_df = pd.DataFrame([input_data])

# --- Ajouter les lags manquants pour LightGBM ---
if model_choice == "LightGBM":
    input_df['Heating_Load_lag_1'] = input_df['Heating_Load_lag1']
    input_df['Heating_Load_lag_2'] = input_df['Heating_Load_lag1']
    input_df['Heating_Load_lag_3'] = input_df['Heating_Load_lag1']
    input_df['Heating_Load_lag_7'] = input_df['Heating_Load_lag1']
    input_df['Cooling_Load_lag_1'] = input_df['Cooling_Load_lag1']

# --- Compléter les colonnes manquantes avec 0 si nécessaire ---
features = features_lgbm if model_choice == "LightGBM" else features_xgb
missing_cols = set(features) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0

# --- Prédictions ---
if model_choice == "LightGBM":
    pred_heat = model_lgbm_heat.predict(input_df[features])[0]
    pred_cool = model_lgbm_cool.predict(input_df[features])[0]
else:  # XGBoost
    pred_heat = model_xgb_heat.predict(input_df[features])[0]
    pred_cool = model_xgb_cool.predict(input_df[features])[0]

# --- Affichage des résultats ---
st.subheader(f"Résultats de la prédiction ({model_choice})")
st.write(f"🔹 Charge de chauffage (Heating Load) : {pred_heat:.2f} kW")
st.write(f"🔹 Charge de refroidissement (Cooling Load) : {pred_cool:.2f} kW")
