# app.py → LightGBM + ARIMA+LSTM + LSTM Multivarié + XGBoost
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==================== CHARGEMENT DES MODÈLES ====================
# LightGBM
model_heat = joblib.load("models/lgbm_heating_model.pkl")
model_cool = joblib.load("models/lgbm_cooling_model.pkl")

# ARIMA+LSTM
arima_heat = joblib.load("models/arima_Heating_Load.pkl")
arima_cool = joblib.load("models/arima_Cooling_Load.pkl")
lstm_heat = load_model("models/lstm_Heating_Load.h5", compile=False)
lstm_cool = load_model("models/lstm_Cooling_Load.h5", compile=False)
scaler_X_heat = joblib.load("models/scaler_X_Heating_Load.pkl")
scaler_X_cool = joblib.load("models/scaler_X_Cooling_Load.pkl")
scaler_resid_heat = joblib.load("models/scaler_resid_Heating_Load.pkl")
scaler_resid_cool = joblib.load("models/scaler_resid_Cooling_Load.pkl")
config_heat = joblib.load("models/model_config_Heating_Load.pkl")
config_cool = joblib.load("models/model_config_Cooling_Load.pkl")
SEQ_LENGTH_HEAT = config_heat["seq_length"]
SEQ_LENGTH_COOL = config_cool["seq_length"]
X_FEATURES_HEAT = config_heat["X_features"]
X_FEATURES_COOL = config_cool["X_features"]

# LSTM Multivarié
lstm_multi_heat = load_model("models/lstm_multi_heat_model.h5", compile=False)
lstm_multi_cool = load_model("models/lstm_multi_cool_model.h5", compile=False)
scaler_multi_heat = joblib.load("models/lstm_multi_heat_scaler.pkl")
scaler_multi_cool = joblib.load("models/lstm_multi_cool_scaler.pkl")
FEATURES_MULTI = [
    'Overall_Height', 'Roof_Area', 'temp', 'irradiation',
    'Surface_Area', 'Relative_Compactness', 'usage_type_encoded',
    'humidity', 'wind_speed', 'Glazing_Area', 'Wall_Area'
]
SEQ_LEN_MULTI = 30
HORIZON = 1  # prédiction d'un jour

# XGBoost
xgb_heat = joblib.load("models/xgboost_Heating_Load.pkl")
xgb_cool = joblib.load("models/xgboost_Cooling_Load.pkl")
scaler_XGBoost = joblib.load("models/scaler_XGBoost_features.pkl")
FEATURES_XGBOOST = [
    'Overall_Height', 'Roof_Area', 'temp', 'irradiation',
    'Surface_Area', 'Relative_Compactness', 'usage_type_encoded',
    'humidity', 'wind_speed', 'Glazing_Area', 'Wall_Area'
]

# ==================== FEATURES EXACTES LightGBM ====================
HEAT_FEATURES = [
    'Overall_Height', 'Roof_Area', 'temp', 'irradiation',
    'Surface_Area', 'Relative_Compactness', 'usage_type_encoded',
    'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution',
    'Heating_Load_lag_1', 'Heating_Load_lag_2', 'Heating_Load_lag_3', 'Heating_Load_lag_7'
]

COOL_FEATURES = [
    'Overall_Height', 'Roof_Area', 'temp', 'irradiation',
    'Surface_Area', 'Relative_Compactness', 'usage_type_encoded',
    'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution',
    'Cooling_Load_lag_1', 'Cooling_Load_lag_2', 'Cooling_Load_lag_3', 'Cooling_Load_lag_7'
]

# ==================== INTERFACE ====================
st.title("Prédiction des charges thermiques")
st.sidebar.header("Choix du modèle")
model_choice = st.sidebar.selectbox(
    "Sélectionnez le modèle",
    ["LightGBM", "ARIMA+LSTM", "LSTM Multivarié", "XGBoost"]
)

# --- Inputs utilisateur ---
Overall_Height = st.sidebar.selectbox("Hauteur totale (m)", [3.5, 7.0])
Roof_Area      = st.sidebar.number_input("Surface toiture (m²)", 100.0, 300.0, 171.0)
Surface_Area   = st.sidebar.number_input("Surface totale (m²)", 500.0, 1000.0, 686.0)
Relative_Compactness = st.sidebar.slider("Compacité relative", 0.60, 1.00, 0.82, 0.01)

# --- Orientation et Répartition du vitrage cachés ---
Orientation = 2  # Nord par défaut
Glazing_Area_Distribution = 1  # Uniforme par défaut

Glazing_Area = st.sidebar.slider("Surface vitrée (m²)", 0.0, 55.0, 20.0, 5.0)
temp = st.sidebar.slider("Température (°C)", -10.0, 40.0, 22.0, 0.5)
irradiation = st.sidebar.slider("Irradiation solaire (W/m²)", 0.0, 1000.0, 450.0, 10.0)
usage_type_encoded = st.sidebar.selectbox("Type de bâtiment",
    options=[0,1,2,3,4], format_func=lambda x: ["Résidentiel","Bureau","Éducatif","Commercial","Industriel"][x])
heating_yest = st.sidebar.number_input("Heating Load hier (kW)", value=18.0, step=0.5)
cooling_yest = st.sidebar.number_input("Cooling Load hier (kW)", value=22.0, step=0.5)

# ==================== PRÉDICTION ====================
if st.button("Prédire pour demain", type="primary"):

    if model_choice == "LightGBM":
        df_heat = pd.DataFrame([{
            'Overall_Height': Overall_Height,
            'Roof_Area': Roof_Area,
            'temp': temp,
            'irradiation': irradiation,
            'Surface_Area': Surface_Area,
            'Relative_Compactness': Relative_Compactness,
            'usage_type_encoded': usage_type_encoded,
            'Orientation': Orientation,
            'Glazing_Area': Glazing_Area,
            'Glazing_Area_Distribution': Glazing_Area_Distribution,
            'Heating_Load_lag_1': heating_yest,
            'Heating_Load_lag_2': heating_yest,
            'Heating_Load_lag_3': heating_yest,
            'Heating_Load_lag_7': heating_yest,
        }]).reindex(columns=HEAT_FEATURES)

        df_cool = pd.DataFrame([{
            'Overall_Height': Overall_Height,
            'Roof_Area': Roof_Area,
            'temp': temp,
            'irradiation': irradiation,
            'Surface_Area': Surface_Area,
            'Relative_Compactness': Relative_Compactness,
            'usage_type_encoded': usage_type_encoded,
            'Orientation': Orientation,
            'Glazing_Area': Glazing_Area,
            'Glazing_Area_Distribution': Glazing_Area_Distribution,
            'Cooling_Load_lag_1': cooling_yest,
            'Cooling_Load_lag_2': cooling_yest,
            'Cooling_Load_lag_3': cooling_yest,
            'Cooling_Load_lag_7': cooling_yest,
        }]).reindex(columns=COOL_FEATURES)

        pred_heat = float(model_heat.predict(df_heat)[0])
        pred_cool = float(model_cool.predict(df_cool)[0])

    elif model_choice == "ARIMA+LSTM":
        # --- Heating ---
        df_input_heat = pd.DataFrame([[Overall_Height, Roof_Area, temp, irradiation,
                                       Surface_Area, Relative_Compactness, usage_type_encoded,
                                       0,0, Glazing_Area,0]], columns=X_FEATURES_HEAT)
        df_scaled_heat = scaler_X_heat.transform(df_input_heat)
        seq_heat = np.zeros((1, SEQ_LENGTH_HEAT, df_scaled_heat.shape[1]+1))
        seq_heat[0, -1, 1:] = df_scaled_heat[-1].reshape(1,-1)
        lstm_pred_scaled_heat = lstm_heat.predict(seq_heat, verbose=0)
        lstm_pred_heat = scaler_resid_heat.inverse_transform(lstm_pred_scaled_heat)[0,0]
        arima_pred_heat = arima_heat.forecast(steps=1).to_numpy()[0]
        pred_heat = float(arima_pred_heat + lstm_pred_heat)

        # --- Cooling ---
        df_input_cool = pd.DataFrame([[Overall_Height, Roof_Area, temp, irradiation,
                                       Surface_Area, Relative_Compactness, usage_type_encoded,
                                       0,0, Glazing_Area,0]], columns=X_FEATURES_COOL)
        df_scaled_cool = scaler_X_cool.transform(df_input_cool)
        seq_cool = np.zeros((1, SEQ_LENGTH_COOL, df_scaled_cool.shape[1]+1))
        seq_cool[0, -1, 1:] = df_scaled_cool[-1].reshape(1,-1)
        lstm_pred_scaled_cool = lstm_cool.predict(seq_cool, verbose=0)
        lstm_pred_cool = scaler_resid_cool.inverse_transform(lstm_pred_scaled_cool)[0,0]
        arima_pred_cool = arima_cool.forecast(steps=1).to_numpy()[0]
        pred_cool = float(arima_pred_cool + lstm_pred_cool)

    elif model_choice == "LSTM Multivarié":
        df_multi = pd.DataFrame([{
            'Overall_Height': Overall_Height,
            'Roof_Area': Roof_Area,
            'temp': temp,
            'irradiation': irradiation,
            'Surface_Area': Surface_Area,
            'Relative_Compactness': Relative_Compactness,
            'usage_type_encoded': usage_type_encoded,
            'humidity': 50,
            'wind_speed': 3,
            'Glazing_Area': Glazing_Area,
            'Wall_Area': 100
        }])[FEATURES_MULTI]

        last_seq_heat = np.vstack([df_multi.values]*SEQ_LEN_MULTI)
        scaled_seq_heat = scaler_multi_heat.transform(np.hstack([last_seq_heat, np.zeros((SEQ_LEN_MULTI,1))]))
        current_seq_heat = scaled_seq_heat[:, :-1]
        lstm_pred_scaled_heat = lstm_multi_heat.predict(current_seq_heat.reshape(1, SEQ_LEN_MULTI, len(FEATURES_MULTI)), verbose=0)
        dummy = np.zeros((1, len(FEATURES_MULTI)+1))
        dummy[0,-1] = lstm_pred_scaled_heat[0,0]
        pred_heat = scaler_multi_heat.inverse_transform(dummy)[:, -1][0]

        last_seq_cool = np.vstack([df_multi.values]*SEQ_LEN_MULTI)
        scaled_seq_cool = scaler_multi_cool.transform(np.hstack([last_seq_cool, np.zeros((SEQ_LEN_MULTI,1))]))
        current_seq_cool = scaled_seq_cool[:, :-1]
        lstm_pred_scaled_cool = lstm_multi_cool.predict(current_seq_cool.reshape(1, SEQ_LEN_MULTI, len(FEATURES_MULTI)), verbose=0)
        dummy = np.zeros((1, len(FEATURES_MULTI)+1))
        dummy[0,-1] = lstm_pred_scaled_cool[0,0]
        pred_cool = scaler_multi_cool.inverse_transform(dummy)[:, -1][0]

    else:  # XGBoost
        df_xgb = pd.DataFrame([{
            'Overall_Height': Overall_Height,
            'Roof_Area': Roof_Area,
            'temp': temp,
            'irradiation': irradiation,
            'Surface_Area': Surface_Area,
            'Relative_Compactness': Relative_Compactness,
            'usage_type_encoded': usage_type_encoded,
            'humidity': 50,
            'wind_speed': 3,
            'Glazing_Area': Glazing_Area,
            'Wall_Area': 100
        }])[FEATURES_XGBOOST]

        df_scaled = scaler_XGBoost.transform(df_xgb)
        pred_heat = float(xgb_heat.predict(df_scaled)[0])
        pred_cool = float(xgb_cool.predict(df_scaled)[0])

    # Affichage
    col1, col2 = st.columns(2)
    col1.metric("Charge de chauffage demain", f"{pred_heat:.2f} kW", delta=f"{pred_heat - heating_yest:+.2f} kW")
    col2.metric("Charge de refroidissement demain", f"{pred_cool:.2f} kW", delta=f"{pred_cool - cooling_yest:+.2f} kW")
    st.success(f"Prédiction réussie avec {model_choice} !")
    st.balloons()
